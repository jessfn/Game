// Ventana + escena 3D con material PBR (metalico/rugosidad), iluminacion dinamica,
// post-procesado realista (bloom, SSAO, MSAA, tonemapping) y un controlador de
// personaje FPS con movimiento fluido (aceleracion, sway, camera bob), arma con
// mira (ADS) y disparo por raycast contra la cobertura del mapa.
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
#include <filament/IndirectLight.h>
#include <filament/Skybox.h>
#include <filament/Options.h>
#include <filament/ColorGrading.h>

#include <utils/EntityManager.h>
#include <geometry/SurfaceOrientation.h>
#include <math/mat4.h>
#include <math/mat3.h>
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

static void addBoxFace(std::vector<Vertex>& verts, std::vector<uint16_t>& indices,
                        float3 center, float3 n, float3 u, float3 v, float3 halfExtents) {
    uint16_t base = (uint16_t)verts.size();
    float3 uu = u * halfExtents;
    float3 vv = v * halfExtents;
    float3 nn = n * halfExtents;
    verts.push_back({ center + nn - uu - vv, n });
    verts.push_back({ center + nn + uu - vv, n });
    verts.push_back({ center + nn + uu + vv, n });
    verts.push_back({ center + nn - uu + vv, n });
    indices.insert(indices.end(), { base, uint16_t(base+1), uint16_t(base+2),
                                     base, uint16_t(base+2), uint16_t(base+3) });
}

// Caja generica (usada para el cubo decorativo y las coberturas del mapa)
static void buildBox(std::vector<Vertex>& verts, std::vector<uint16_t>& indices,
                      float3 center, float3 halfExtents) {
    addBoxFace(verts, indices, center, { 0,  0,  1}, {1,0,0}, {0,1,0}, halfExtents);
    addBoxFace(verts, indices, center, { 0,  0, -1}, {-1,0,0}, {0,1,0}, halfExtents);
    addBoxFace(verts, indices, center, { 1,  0,  0}, {0,0,-1}, {0,1,0}, halfExtents);
    addBoxFace(verts, indices, center, {-1,  0,  0}, {0,0,1}, {0,1,0}, halfExtents);
    addBoxFace(verts, indices, center, { 0,  1,  0}, {1,0,0}, {0,0,-1}, halfExtents);
    addBoxFace(verts, indices, center, { 0, -1,  0}, {1,0,0}, {0,0,1}, halfExtents);
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

static void makeMesh(
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

    auto* tangents = new std::vector<short4>(verts.size());
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
}

// Cobertura estatica del mapa: caja con centro/mitad-de-extension, usada tanto
// para el render como para la colision del jugador y el raycast de disparo.
struct CoverBox {
    float3 center;
    float3 halfExtents;
};

static bool rayIntersectsBox(float3 origin, float3 dir, const CoverBox& box, float& tHit) {
    float3 minB = box.center - box.halfExtents;
    float3 maxB = box.center + box.halfExtents;
    float tmin = 0.0f, tmax = 1000.0f;
    for (int i = 0; i < 3; ++i) {
        float o = (i == 0 ? origin.x : i == 1 ? origin.y : origin.z);
        float d = (i == 0 ? dir.x : i == 1 ? dir.y : dir.z);
        float mn = (i == 0 ? minB.x : i == 1 ? minB.y : minB.z);
        float mx = (i == 0 ? maxB.x : i == 1 ? maxB.y : maxB.z);
        if (std::abs(d) < 1e-6f) {
            if (o < mn || o > mx) return false;
        } else {
            float t1 = (mn - o) / d;
            float t2 = (mx - o) / d;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return false;
        }
    }
    tHit = tmin;
    return true;
}

// Resuelve el movimiento del jugador contra las coberturas (circulo horizontal vs caja expandida)
static float3 resolveCollisions(float3 pos, const std::vector<CoverBox>& covers, float playerRadius) {
    for (const auto& box : covers) {
        float3 minB = box.center - box.halfExtents - float3{playerRadius, 0, playerRadius};
        float3 maxB = box.center + box.halfExtents + float3{playerRadius, 0, playerRadius};
        if (pos.x > minB.x && pos.x < maxB.x && pos.z > minB.z && pos.z < maxB.z) {
            float penLeft   = pos.x - minB.x;
            float penRight  = maxB.x - pos.x;
            float penFront  = pos.z - minB.z;
            float penBack   = maxB.z - pos.z;
            float minPen = std::min({penLeft, penRight, penFront, penBack});
            if (minPen == penLeft) pos.x = minB.x;
            else if (minPen == penRight) pos.x = maxB.x;
            else if (minPen == penFront) pos.z = minB.z;
            else pos.z = maxB.z;
        }
    }
    return pos;
}

// --- Estado del controlador de personaje FPS ---
struct PlayerController {
    float3 position{0.0f, 1.8f, 10.0f};
    float3 velocityXZ{0, 0, 0}; // velocidad horizontal suavizada (aceleracion/friccion)
    float yaw = -kPi / 2.0f;
    float pitch = 0.0f;

    float verticalVelocity = 0.0f;
    bool onGround = true;
    bool crouching = false;
    bool aiming = false;

    float bobTime = 0.0f;
    float bobAmount = 0.0f; // 0..1 segun cuanto se mueve el jugador

    static constexpr float kEyeHeightStand = 1.8f;
    static constexpr float kEyeHeightCrouch = 1.1f;
    static constexpr float kWalkSpeed = 4.2f;
    static constexpr float kSprintSpeed = 7.8f;
    static constexpr float kCrouchSpeed = 2.0f;
    static constexpr float kAimSpeedMul = 0.55f;
    static constexpr float kAcceleration = 22.0f;
    static constexpr float kFriction = 14.0f;
    static constexpr float kJumpVelocity = 5.2f;
    static constexpr float kGravity = -15.5f;
    static constexpr float kGroundY = 0.0f;
    static constexpr float kPlayerRadius = 0.4f;

    float currentEyeHeight = kEyeHeightStand;
    float currentFov = 70.0;

    void update(GLFWwindow* window, float dt, float mouseDx, float mouseDy,
                const std::vector<CoverBox>& covers) {
        aiming = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

        // --- Mirar con el mouse (mas fina al apuntar, como en consola con "aim assist" de sensibilidad) ---
        float sensitivity = aiming ? 0.0012f : 0.0025f;
        yaw += mouseDx * sensitivity;
        pitch -= mouseDy * sensitivity;
        const float pitchLimit = kPi / 2.0f - 0.01f;
        pitch = std::clamp(pitch, -pitchLimit, pitchLimit);

        float3 forward{ std::cos(pitch) * std::cos(yaw), 0.0f, std::cos(pitch) * std::sin(yaw) };
        forward = normalize(forward);
        float3 right = normalize(cross(forward, float3{0, 1, 0}));

        bool crouchHeld = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                           glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS;
        crouching = crouchHeld;

        bool sprintHeld = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS && !crouching && !aiming;

        float targetSpeed = crouching ? kCrouchSpeed : (sprintHeld ? kSprintSpeed : kWalkSpeed);
        if (aiming) targetSpeed *= kAimSpeedMul;

        float3 wishDir{0, 0, 0};
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) wishDir += forward;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) wishDir -= forward;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) wishDir += right;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) wishDir -= right;
        bool moving = dot(wishDir, wishDir) > 0.0001f;
        if (moving) wishDir = normalize(wishDir);

        // --- Aceleracion/friccion en vez de velocidad instantanea: se siente fluido, no "robotico" ---
        float3 targetVel = wishDir * targetSpeed;
        float3 diff = targetVel - velocityXZ;
        float accel = moving ? kAcceleration : kFriction;
        float maxDelta = accel * dt;
        float diffLen = std::sqrt(dot(diff, diff));
        if (diffLen > maxDelta && diffLen > 0.0001f) {
            velocityXZ += diff * (maxDelta / diffLen);
        } else {
            velocityXZ = targetVel;
        }

        float3 horizontalMove = velocityXZ * dt;
        float3 newPos = position + horizontalMove;
        newPos = resolveCollisions(newPos, covers, kPlayerRadius);
        position.x = newPos.x;
        position.z = newPos.z;

        // --- Salto y gravedad ---
        if (onGround && glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && !crouching) {
            verticalVelocity = kJumpVelocity;
            onGround = false;
        }
        verticalVelocity += kGravity * dt;

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

        const float mapHalf = 28.0f;
        position.x = std::clamp(position.x, -mapHalf, mapHalf);
        position.z = std::clamp(position.z, -mapHalf, mapHalf);

        // --- Camera bob: se activa con la velocidad horizontal real, se apaga al apuntar ---
        float speedFrac = std::min(1.0f, std::sqrt(dot(velocityXZ, velocityXZ)) / kSprintSpeed);
        float targetBob = onGround ? (aiming ? speedFrac * 0.15f : speedFrac) : 0.0f;
        bobAmount += (targetBob - bobAmount) * std::min(1.0f, dt * 8.0f);
        if (onGround && speedFrac > 0.05f) {
            bobTime += dt * (sprintHeld ? 11.0f : 8.0f);
        }

        // --- FOV: se cierra al apuntar (mira), simulando zoom del arma ---
        double targetFov = aiming ? 40.0 : 70.0;
        currentFov += (targetFov - currentFov) * std::min(1.0, double(dt) * 12.0);
    }

    float3 eyePosition() const {
        float bobY = std::sin(bobTime * 2.0f) * 0.045f * bobAmount;
        float bobX = std::sin(bobTime) * 0.03f * bobAmount;
        return position + float3{bobX, bobY, 0};
    }

    float3 lookTarget() const {
        float3 dir{ std::cos(pitch) * std::cos(yaw), std::sin(pitch), std::cos(pitch) * std::sin(yaw) };
        return eyePosition() + dir;
    }

    float3 forwardVector() const {
        return normalize(float3{ std::cos(pitch) * std::cos(yaw), std::sin(pitch), std::cos(pitch) * std::sin(yaw) });
    }
};

int main() {
    if (!glfwInit()) {
        std::cerr << "Fallo glfwInit" << std::endl;
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    const int width = 1600, height = 900;
    GLFWwindow* window = glfwCreateWindow(width, height, "Mi Juego - FPS", nullptr, nullptr);
    if (!window) {
        std::cerr << "Fallo al crear ventana" << std::endl;
        return 1;
    }

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
    camera->setProjection(70.0, double(width) / height, 0.05, 1000.0, Camera::Fov::VERTICAL);

    View* view = engine->createView();
    view->setCamera(camera);
    view->setScene(scene);
    view->setViewport({0, 0, (uint32_t)width, (uint32_t)height});

    // --- Post-procesado realista: tonemap filmico, bloom, SSAO, MSAA y dithering ---
    ColorGrading* colorGrading = ColorGrading::Builder()
        .toneMapping(ColorGrading::ToneMapping::ACES)
        .build(*engine);
    view->setColorGrading(colorGrading);
    view->setDithering(View::Dithering::TEMPORAL);

    BloomOptions bloom;
    bloom.enabled = true;
    bloom.strength = 0.12f;
    view->setBloomOptions(bloom);

    AmbientOcclusionOptions ao;
    ao.enabled = true;
    ao.radius = 0.5f;
    ao.power = 1.2f;
    view->setAmbientOcclusionOptions(ao);

    // MSAA deshabilitado a proposito: en GPUs integradas con drivers viejos, combinar
    // MSAA con SSAO/Bloom puede colgar el driver (TDR de Windows) y se ve como "se queda
    // trabado en blanco". FXAA por si sola ya suaviza bordes con costo minimo.
    view->setAntiAliasing(View::AntiAliasing::FXAA);

    // --- Luz ambiental (cielo) plana, para que las sombras no queden negras ---
    float3 skyIrradiance[1] = { float3{0.55f, 0.62f, 0.72f} * 25000.0f };
    IndirectLight* ibl = IndirectLight::Builder()
        .irradiance(1, skyIrradiance)
        .intensity(25000.0f)
        .build(*engine);
    scene->setIndirectLight(ibl);

    Skybox* skybox = Skybox::Builder()
        .color({0.45f, 0.55f, 0.70f, 1.0f})
        .build(*engine);
    scene->setSkybox(skybox);

    // --- Material PBR compilado por matc en tiempo de build (assets/lit.filamat) ---
    std::vector<uint8_t> matData = readFile("assets/lit.filamat");
    Material* material = Material::Builder()
        .package(matData.data(), matData.size())
        .build(*engine);

    auto makeMat = [&](float3 color, float metallic, float roughness) {
        MaterialInstance* inst = material->createInstance();
        inst->setParameter("baseColor", RgbType::LINEAR, color);
        inst->setParameter("metallic", metallic);
        inst->setParameter("roughness", roughness);
        return inst;
    };

    MaterialInstance* cubeMat = makeMat({0.7f, 0.1f, 0.1f}, 0.3f, 0.35f);
    MaterialInstance* groundMat = makeMat({0.22f, 0.24f, 0.20f}, 0.0f, 0.95f);
    MaterialInstance* coverMat = makeMat({0.35f, 0.33f, 0.30f}, 0.05f, 0.75f);
    MaterialInstance* weaponMat = makeMat({0.06f, 0.06f, 0.07f}, 0.6f, 0.3f);
    MaterialInstance* muzzleFlashMat = makeMat({1.0f, 0.75f, 0.25f}, 0.0f, 1.0f);

    // --- Cubo de referencia ---
    std::vector<Vertex> cubeVerts;
    std::vector<uint16_t> cubeIndices;
    buildBox(cubeVerts, cubeIndices, {0, 0, 0}, {1, 1, 1});
    utils::Entity cubeRenderable = utils::EntityManager::get().create();
    makeMesh(engine, cubeRenderable, cubeMat, cubeVerts, cubeIndices, {{-1,-1,-1},{1,1,1}});
    scene->addEntity(cubeRenderable);

    // --- Suelo (mapa) ---
    const float mapHalfSize = 28.0f;
    std::vector<Vertex> groundVerts;
    std::vector<uint16_t> groundIndices;
    buildGround(groundVerts, groundIndices, mapHalfSize);
    utils::Entity groundRenderable = utils::EntityManager::get().create();
    makeMesh(engine, groundRenderable, groundMat, groundVerts, groundIndices,
             {{-mapHalfSize, -0.01f, -mapHalfSize}, {mapHalfSize, 0.01f, mapHalfSize}});
    scene->addEntity(groundRenderable);

    // --- Coberturas del mapa (cajas para moverse, agacharse y cubrirse) ---
    std::vector<CoverBox> covers = {
        { {  6.0f, 0.9f,   2.0f}, {1.2f, 0.9f, 1.2f} },
        { { -6.0f, 0.9f,  -3.0f}, {1.2f, 0.9f, 1.2f} },
        { {  0.0f, 0.6f, -10.0f}, {3.0f, 0.6f, 0.6f} },
        { { 10.0f, 1.5f, -6.0f}, {0.8f, 1.5f, 4.0f} },
        { {-10.0f, 1.5f,  6.0f}, {0.8f, 1.5f, 4.0f} },
        { {  0.0f, 2.0f,  16.0f}, {5.0f, 2.0f, 0.6f} },
    };
    std::vector<utils::Entity> coverEntities;
    for (auto& box : covers) {
        std::vector<Vertex> bv; std::vector<uint16_t> bi;
        buildBox(bv, bi, box.center, box.halfExtents);
        utils::Entity ent = utils::EntityManager::get().create();
        makeMesh(engine, ent, coverMat, bv, bi,
                 {{ box.center.x - box.halfExtents.x, box.center.y - box.halfExtents.y, box.center.z - box.halfExtents.z },
                  { box.center.x + box.halfExtents.x, box.center.y + box.halfExtents.y, box.center.z + box.halfExtents.z }});
        scene->addEntity(ent);
        coverEntities.push_back(ent);
    }

    // --- Luz direccional (sol) ---
    utils::Entity sun = utils::EntityManager::get().create();
    LightManager::Builder(LightManager::Type::SUN)
        .color(Color::toLinear<ACCURATE>(sRGBColor{0.98f, 0.92f, 0.89f}))
        .intensity(110000)
        .direction({0.5f, -1.0f, -0.3f})
        .sunAngularRadius(0.9f)
        .sunHaloSize(10.0f)
        .sunHaloFalloff(80.0f)
        .castShadows(true)
        .build(*engine, sun);
    scene->addEntity(sun);

    // --- Arma en primera persona (view model): cuerpo + cano, hijos de la camara logica ---
    utils::Entity weaponBody = utils::EntityManager::get().create();
    {
        std::vector<Vertex> wv; std::vector<uint16_t> wi;
        buildBox(wv, wi, {0, 0, 0}, {0.05f, 0.06f, 0.35f});
        makeMesh(engine, weaponBody, weaponMat, wv, wi, {{-0.05f,-0.06f,-0.35f},{0.05f,0.06f,0.35f}});
        scene->addEntity(weaponBody);
    }
    utils::Entity weaponBarrel = utils::EntityManager::get().create();
    {
        std::vector<Vertex> wv; std::vector<uint16_t> wi;
        buildBox(wv, wi, {0, 0, 0}, {0.025f, 0.025f, 0.22f});
        makeMesh(engine, weaponBarrel, weaponMat, wv, wi, {{-0.025f,-0.025f,-0.22f},{0.025f,0.025f,0.22f}});
        scene->addEntity(weaponBarrel);
    }

    // --- Luz de disparo (muzzle flash), apagada por defecto ---
    utils::Entity muzzleLight = utils::EntityManager::get().create();
    LightManager::Builder(LightManager::Type::POINT)
        .color({1.0f, 0.75f, 0.35f})
        .intensity(0.0f)
        .falloff(4.0f)
        .build(*engine, muzzleLight);
    scene->addEntity(muzzleLight);

    utils::Entity muzzleFlashMesh = utils::EntityManager::get().create();
    {
        std::vector<Vertex> wv; std::vector<uint16_t> wi;
        buildBox(wv, wi, {0, 0, 0}, {0.06f, 0.06f, 0.06f});
        makeMesh(engine, muzzleFlashMesh, muzzleFlashMat, wv, wi, {{-0.06f,-0.06f,-0.06f},{0.06f,0.06f,0.06f}});
        scene->addEntity(muzzleFlashMesh);
    }

    PlayerController player;

    double lastMouseX = width / 2.0, lastMouseY = height / 2.0;
    glfwSetCursorPos(window, lastMouseX, lastMouseY);
    double lastTime = glfwGetTime();

    bool prevFirePressed = false;
    float fireCooldown = 0.0f;
    float muzzleFlashTimer = 0.0f;
    static constexpr float kFireRate = 0.11f; // ~9 disparos/seg, tipo rifle automatico

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

        player.update(window, dt, dx, dy, covers);

        camera->setProjection(player.currentFov, double(width) / height, 0.05, 1000.0, Camera::Fov::VERTICAL);
        float3 eye = player.eyePosition();
        camera->lookAt(eye, player.lookTarget(), {0, 1, 0});

        // --- Disparo: raycast desde el ojo hacia adelante contra las coberturas ---
        fireCooldown = std::max(0.0f, fireCooldown - dt);
        bool firePressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        if (firePressed && fireCooldown <= 0.0f) {
            fireCooldown = kFireRate;
            muzzleFlashTimer = 0.05f;
            float3 dir = player.forwardVector();
            float closestT = 1e9f;
            bool hit = false;
            for (auto& box : covers) {
                float t;
                if (rayIntersectsBox(eye, dir, box, t) && t < closestT) { closestT = t; hit = true; }
            }
            if (hit) {
                std::cout << "Impacto en cobertura a " << closestT << "m" << std::endl;
            } else {
                std::cout << "Disparo al aire" << std::endl;
            }
        }
        prevFirePressed = firePressed;
        muzzleFlashTimer = std::max(0.0f, muzzleFlashTimer - dt);

        // --- Posicionar el arma relativa a la camara (view model), con sway y ADS ---
        float3 fwd = player.forwardVector();
        float3 right = normalize(cross(fwd, float3{0, 1, 0}));
        float3 up = cross(right, fwd);

        float3 hipOffset = right * 0.28f - up * 0.22f + fwd * 0.55f;
        float3 adsOffset = fwd * 0.45f - up * 0.02f;
        float aimT = player.aiming ? 1.0f : 0.0f;
        static float aimBlend = 0.0f;
        aimBlend += (aimT - aimBlend) * std::min(1.0f, dt * 12.0f);
        float3 weaponOffset = hipOffset * (1.0f - aimBlend) + adsOffset * aimBlend;

        float swayX = std::sin(player.bobTime) * 0.01f * player.bobAmount * (1.0f - aimBlend * 0.7f);
        float swayY = std::sin(player.bobTime * 2.0f) * 0.008f * player.bobAmount * (1.0f - aimBlend * 0.7f);
        float3 weaponPos = eye + weaponOffset + right * swayX + up * swayY;

        auto& tcm = engine->getTransformManager();
        // Construimos la orientacion del arma directamente desde forward/right/up para evitar gimbal issues
        mat3f basis(right, up, -fwd);
        mat4f weaponTransform(basis, weaponPos);

        tcm.setTransform(tcm.getInstance(weaponBody), weaponTransform);
        tcm.setTransform(tcm.getInstance(weaponBarrel),
            weaponTransform * mat4f::translation(float3{0, 0.02f, -0.5f}));

        float3 muzzlePos = weaponPos + fwd * (0.55f + 0.22f) - up * 0.0f;
        auto& lcm = engine->getLightManager();
        lcm.setIntensity(lcm.getInstance(muzzleLight), muzzleFlashTimer > 0.0f ? 2000.0f : 0.0f);
        lcm.setPosition(lcm.getInstance(muzzleLight), muzzlePos);

        float flashScale = muzzleFlashTimer > 0.0f ? 1.0f : 0.0001f;
        tcm.setTransform(tcm.getInstance(muzzleFlashMesh),
            mat4f::translation(muzzlePos) * mat4f::scaling(float3{flashScale}));

        // El cubo decorativo gira lentamente para mostrar el sombreado PBR
        static float angle = 0.0f;
        angle += 0.5f * dt;
        tcm.setTransform(tcm.getInstance(cubeRenderable),
            mat4f::translation(float3{0, 1.0f, -4.0f}) * mat4f::rotation(angle, float3{0, 1, 0}));

        if (renderer->beginFrame(swapChain)) {
            renderer->render(view);
            renderer->endFrame();
        }
    }

    engine->destroy(sun);
    engine->destroy(muzzleLight);
    engine->destroy(muzzleFlashMesh);
    engine->destroy(weaponBody);
    engine->destroy(weaponBarrel);
    engine->destroy(cubeRenderable);
    engine->destroy(groundRenderable);
    for (auto& ent : coverEntities) engine->destroy(ent);
    engine->destroy(cubeMat);
    engine->destroy(groundMat);
    engine->destroy(coverMat);
    engine->destroy(weaponMat);
    engine->destroy(muzzleFlashMat);
    engine->destroy(material);
    engine->destroy(skybox);
    engine->destroy(ibl);
    engine->destroy(colorGrading);
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
