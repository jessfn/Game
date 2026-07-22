// Ventana + escena 3D con material PBR (metalico/rugosidad), iluminacion dinamica
// y un controlador de personaje FPS (caminar, correr, saltar, agacharse, mouse-look).
// Motor de render: Filament (Google). Ventana/input: GLFW. Todo corre nativo, sin editor.

#include <filament/Engine.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/View.h>
#include <filament/Camera.h>
#include <filament/SwapChain.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/VertexBuffer.h>
#include <filament/IndexBuffer.h>
#include <filament/RenderableManager.h>
#include <filament/TransformManager.h>
#include <filament/LightManager.h>
#include <filament/Viewport.h>

#include <utils/EntityManager.h>
#include <geometry/SurfaceOrientation.h>
#include <math/mat4.h>
#include <math/vec3.h>
#include <math/norm.h>

#define NOMINMAX
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace filament;
using namespace filament::math;

static constexpr float kPi = 3.14159265358979323846f;

// Carga un archivo binario completo (usado para el .filamat compilado)
static std::vector<uint8_t> readFile(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "No se pudo abrir: " << path << std::endl;
        std::exit(1);
    }
    size_t size = (size_t)f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(size);
    f.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

struct Vertex { float3 position; float3 normal; };

// Geometria de un cubo con posiciones y normales por cara (para sombreado PBR correcto)
static void buildCube(std::vector<Vertex>& verts, std::vector<uint16_t>& indices) {
    const float3 facesNormals[6] = {
        { 0,  0,  1}, { 0,  0, -1},
        { 1,  0,  0}, {-1,  0,  0},
        { 0,  1,  0}, { 0, -1,  0}
    };
    auto addFace = [&](float3 n, float3 u, float3 v) {
        uint16_t base = (uint16_t)verts.size();
        verts.push_back({ n * 0.5f - u * 0.5f - v * 0.5f, n });
        verts.push_back({ n * 0.5f + u * 0.5f - v * 0.5f, n });
        verts.push_back({ n * 0.5f + u * 0.5f + v * 0.5f, n });
        verts.push_back({ n * 0.5f - u * 0.5f + v * 0.5f, n });
        indices.insert(indices.end(), { base, uint16_t(base+1), uint16_t(base+2),
                                         base, uint16_t(base+2), uint16_t(base+3) });
    };
    addFace(facesNormals[0], {1,0,0}, {0,1,0});
    addFace(facesNormals[1], {-1,0,0}, {0,1,0});
    addFace(facesNormals[2], {0,0,-1}, {0,1,0});
    addFace(facesNormals[3], {0,0,1}, {0,1,0});
    addFace(facesNormals[4], {1,0,0}, {0,0,-1});
    addFace(facesNormals[5], {1,0,0}, {0,0,1});
}

// Plano grande y plano en Y=0 (el "suelo" del mapa), con normales hacia arriba
static void buildGround(std::vector<Vertex>& verts, std::vector<uint16_t>& indices, float halfSize) {
    float3 n = {0, 1, 0};
    verts.push_back({ {-halfSize, 0, -halfSize}, n });
    verts.push_back({ { halfSize, 0, -halfSize}, n });
    verts.push_back({ { halfSize, 0,  halfSize}, n });
    verts.push_back({ {-halfSize, 0,  halfSize}, n });
    indices.insert(indices.end(), { 0, 1, 2, 0, 2, 3 });
}

static RenderableManager::Builder::Result makeMesh(
    Engine* engine, utils::Entity entity, MaterialInstance* matInstance,
    const std::vector<Vertex>& verts, const std::vector<uint16_t>& indices,
    const filament::Box& bbox)
{
    std::vector<float3> normals;
    normals.reserve(verts.size());
    for (auto& v : verts) normals.push_back(v.normal);

    geometry::SurfaceOrientation* orientation = geometry::SurfaceOrientation::Builder()
        .vertexCount(verts.size())
        .normals(normals.data())
        .build();

    std::vector<short4>* tangents = new std::vector<short4>(verts.size());
    orientation->getQuats(tangents->data(), verts.size());
    delete orientation;

    VertexBuffer* vb = VertexBuffer::Builder()
        .vertexCount((uint32_t)verts.size())
        .bufferCount(2)
        .attribute(VertexAttribute::POSITION, 0, VertexBuffer::AttributeType::FLOAT3, 0, sizeof(Vertex))
        .attribute(VertexAttribute::TANGENTS, 1, VertexBuffer::AttributeType::SHORT4, 0, sizeof(short4))
        .normalized(VertexAttribute::TANGENTS)
        .build(*engine);

    vb->setBufferAt(*engine, 0, VertexBuffer::BufferDescriptor(
        verts.data(), verts.size() * sizeof(Vertex), nullptr));
    vb->setBufferAt(*engine, 1, VertexBuffer::BufferDescriptor(
        tangents->data(), tangents->size() * sizeof(short4),
        [](void*, size_t, void* user) { delete static_cast<std::vector<short4>*>(user); },
        tangents));

    IndexBuffer* ib = IndexBuffer::Builder()
        .indexCount((uint32_t)indices.size())
        .bufferType(IndexBuffer::IndexType::USHORT)
        .build(*engine);
    ib->setBuffer(*engine, IndexBuffer::BufferDescriptor(
        indices.data(), indices.size() * sizeof(uint16_t), nullptr));

    RenderableManager::Builder(1)
        .boundingBox(bbox)
        .material(0, matInstance)
        .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, vb, ib)
        .culling(false)
        .build(*engine, entity);

    return {};
}

// --- Estado del controlador de personaje FPS ---
struct PlayerController {
    float3 position{0.0f, 1.8f, 6.0f}; // altura de ojos de pie
    float yaw = -kPi / 2.0f;   // mirando hacia -Z
    float pitch = 0.0f;

    float verticalVelocity = 0.0f;
    bool onGround = true;
    bool crouching = false;

    static constexpr float kEyeHeightStand = 1.8f;
    static constexpr float kEyeHeightCrouch = 1.1f;
    static constexpr float kWalkSpeed = 4.0f;
    static constexpr float kSprintSpeed = 7.5f;
    static constexpr float kCrouchSpeed = 2.0f;
    static constexpr float kJumpVelocity = 5.0f;
    static constexpr float kGravity = -15.0f;
    static constexpr float kGroundY = 0.0f;

    float currentEyeHeight = kEyeHeightStand;

    void update(GLFWwindow* window, float dt, float mouseDx, float mouseDy) {
        // --- Mirar con el mouse ---
        const float sensitivity = 0.0025f;
        yaw += mouseDx * sensitivity;
        pitch -= mouseDy * sensitivity;
        const float pitchLimit = kPi / 2.0f - 0.01f;
        pitch = std::clamp(pitch, -pitchLimit, pitchLimit);

        // --- Direcciones de movimiento en el plano horizontal ---
        float3 forward{ std::cos(pitch) * std::cos(yaw), 0.0f, std::cos(pitch) * std::sin(yaw) };
        forward = normalize(forward);
        float3 right = normalize(cross(forward, float3{0, 1, 0}));

        bool crouchHeld = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                           glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS;
        crouching = crouchHeld;

        bool sprintHeld = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS && !crouching;

        float speed = crouching ? kCrouchSpeed : (sprintHeld ? kSprintSpeed : kWalkSpeed);

        float3 moveDir{0, 0, 0};
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) moveDir += forward;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) moveDir -= forward;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) moveDir += right;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) moveDir -= right;

        if (dot(moveDir, moveDir) > 0.0001f) {
            moveDir = normalize(moveDir);
            position += moveDir * speed * dt;
        }

        // --- Salto y gravedad ---
        if (onGround && glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && !crouching) {
            verticalVelocity = kJumpVelocity;
            onGround = false;
        }

        verticalVelocity += kGravity * dt;

        // --- Altura de ojos: interpolacion suave al agacharse ---
        float targetEyeHeight = crouching ? kEyeHeightCrouch : kEyeHeightStand;
        currentEyeHeight += (targetEyeHeight - currentEyeHeight) * std::min(1.0f, dt * 10.0f);

        float feetY = position.y - currentEyeHeight + verticalVelocity * dt;
        if (feetY <= kGroundY) {
            feetY = kGroundY;
            verticalVelocity = 0.0f;
            onGround = true;
        } else {
            onGround = false;
        }
        position.y = feetY + currentEyeHeight;

        // --- Limites del mapa (mismo tamano que el suelo) ---
        const float mapHalf = 24.0f;
        position.x = std::clamp(position.x, -mapHalf, mapHalf);
        position.z = std::clamp(position.z, -mapHalf, mapHalf);
    }

    float3 lookTarget() const {
        float3 dir{ std::cos(pitch) * std::cos(yaw), std::sin(pitch), std::cos(pitch) * std::sin(yaw) };
        return position + dir;
    }
};

int main() {
    if (!glfwInit()) {
        std::cerr << "Fallo glfwInit" << std::endl;
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    const int width = 1280, height = 720;
    GLFWwindow* window = glfwCreateWindow(width, height, "Mi Juego - FPS", nullptr, nullptr);
    if (!window) {
        std::cerr << "Fallo al crear ventana" << std::endl;
        return 1;
    }

    // Capturar el mouse para el control de camara estilo shooter
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    void* nativeWindow = glfwGetWin32Window(window);

    Engine* engine = Engine::create(Engine::Backend::OPENGL);
    SwapChain* swapChain = engine->createSwapChain(nativeWindow);
    Renderer* renderer = engine->createRenderer();

    Scene* scene = engine->createScene();

    utils::Entity cameraEntity = utils::EntityManager::get().create();
    Camera* camera = engine->createCamera(cameraEntity);
    camera->setProjection(70.0, double(width) / height, 0.1, 1000.0, Camera::Fov::VERTICAL);

    View* view = engine->createView();
    view->setCamera(camera);
    view->setScene(scene);
    view->setViewport({0, 0, (uint32_t)width, (uint32_t)height});

    // --- Material PBR compilado por matc en tiempo de build (assets/lit.filamat) ---
    std::vector<uint8_t> matData = readFile("assets/lit.filamat");
    Material* material = Material::Builder()
        .package(matData.data(), matData.size())
        .build(*engine);

    MaterialInstance* cubeMat = material->createInstance();
    cubeMat->setParameter("baseColor", RgbType::LINEAR, float3{0.7f, 0.1f, 0.1f});
    cubeMat->setParameter("metallic", 0.2f);
    cubeMat->setParameter("roughness", 0.4f);

    MaterialInstance* groundMat = material->createInstance();
    groundMat->setParameter("baseColor", RgbType::LINEAR, float3{0.25f, 0.27f, 0.24f});
    groundMat->setParameter("metallic", 0.0f);
    groundMat->setParameter("roughness", 0.95f);

    // --- Cubo de referencia ---
    std::vector<Vertex> cubeVerts;
    std::vector<uint16_t> cubeIndices;
    buildCube(cubeVerts, cubeIndices);
    utils::Entity cubeRenderable = utils::EntityManager::get().create();
    makeMesh(engine, cubeRenderable, cubeMat, cubeVerts, cubeIndices,
             {{-1, -1, -1}, {1, 1, 1}});
    scene->addEntity(cubeRenderable);

    // Elevar el cubo para que no quede enterrado en el suelo
    {
        auto& tcm = engine->getTransformManager();
        auto inst = tcm.getInstance(cubeRenderable);
        tcm.setTransform(inst, mat4f::translation(float3{0, 1.0f, 0}));
    }

    // --- Suelo (mapa) ---
    const float mapHalfSize = 24.0f;
    std::vector<Vertex> groundVerts;
    std::vector<uint16_t> groundIndices;
    buildGround(groundVerts, groundIndices, mapHalfSize);
    utils::Entity groundRenderable = utils::EntityManager::get().create();
    makeMesh(engine, groundRenderable, groundMat, groundVerts, groundIndices,
             {{-mapHalfSize, -0.01f, -mapHalfSize}, {mapHalfSize, 0.01f, mapHalfSize}});
    scene->addEntity(groundRenderable);

    // --- Luz direccional (sol) ---
    utils::Entity light = utils::EntityManager::get().create();
    LightManager::Builder(LightManager::Type::SUN)
        .color(Color::toLinear<ACCURATE>(sRGBColor{0.98f, 0.92f, 0.89f}))
        .intensity(110000)
        .direction({0.5f, -1.0f, -0.3f})
        .castShadows(true)
        .build(*engine, light);
    scene->addEntity(light);

    // --- Controlador de jugador (FPS: caminar, correr, saltar, agacharse) ---
    PlayerController player;
    camera->lookAt(player.position, player.lookTarget(), {0, 1, 0});

    double lastMouseX = width / 2.0, lastMouseY = height / 2.0;
    glfwSetCursorPos(window, lastMouseX, lastMouseY);

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        double now = glfwGetTime();
        float dt = std::min(0.05f, float(now - lastTime));
        lastTime = now;

        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        float dx = float(mx - lastMouseX);
        float dy = float(my - lastMouseY);
        lastMouseX = mx;
        lastMouseY = my;

        player.update(window, dt, dx, dy);
        camera->lookAt(player.position, player.lookTarget(), {0, 1, 0});

        // El cubo gira lentamente para mostrar el sombreado PBR
        auto& tcm = engine->getTransformManager();
        auto instance = tcm.getInstance(cubeRenderable);
        static float angle = 0.0f;
        angle += 0.5f * dt;
        tcm.setTransform(instance,
            mat4f::translation(float3{0, 1.0f, 0}) * mat4f::rotation(angle, float3{0, 1, 0}));

        if (renderer->beginFrame(swapChain)) {
            renderer->render(view);
            renderer->endFrame();
        }
    }

    engine->destroy(light);
    engine->destroy(cubeRenderable);
    engine->destroy(groundRenderable);
    engine->destroy(cubeMat);
    engine->destroy(groundMat);
    engine->destroy(material);
    engine->destroy(view);
    engine->destroy(scene);
    engine->destroyCameraComponent(cameraEntity);
    engine->destroy(swapChain);
    engine->destroy(renderer);
    Engine::destroy(&engine);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
