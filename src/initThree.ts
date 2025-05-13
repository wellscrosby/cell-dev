import * as THREE from 'three';
import Stats from "three/addons/libs/stats.module.js";
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VolumeRenderShader1 } from 'three/addons/shaders/VolumeShader.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { DiffusionSim } from './processVolume';

// Global variables
let stats: Stats;
let scene: THREE.Scene;
let renderer: THREE.WebGLRenderer;
let camera: THREE.OrthographicCamera;
let controls: OrbitControls;
let volumeTexture: THREE.Data3DTexture;
let mesh: THREE.Mesh;
let dimensions = { x: 128, y: 128, z: 128 };
let isPlaying = false;
let gui: GUI;
let diffusionSim: DiffusionSim;
let simIterationsPerFrame = 10;
let diffusionConstant = 1.0;
let spheres: THREE.Mesh[] = [];
let axesHelper: THREE.AxesHelper;

function createStats() {
    const stats = new Stats();
    document.body.appendChild(stats.dom);
    return stats;
}

function createScene() {
    const scene = new THREE.Scene();
    const skyboxColor = new THREE.Color(0xf7edd5); // Changed to black background
    scene.background = skyboxColor;
    const minDimension = Math.min(dimensions.x, dimensions.y, dimensions.z);
    axesHelper = new THREE.AxesHelper(minDimension / 3);
    axesHelper.setColors(
        new THREE.Color(0xff0000).multiplyScalar(2), // Red x-axis
        new THREE.Color(0x00ff00).multiplyScalar(2), // Green y-axis 
        new THREE.Color(0x0000ff).multiplyScalar(2)  // Blue z-axis
    );
    axesHelper.position.set(-dimensions.x / 20, -dimensions.y / 20, - dimensions.z / 20);

    scene.add(axesHelper);
    return scene;
}

function updateAxesHelper() {
    scene.remove(axesHelper);
    const avgDimension = (dimensions.x + dimensions.y + dimensions.z) / 3;
    axesHelper = new THREE.AxesHelper(avgDimension / 3);
    axesHelper.setColors(
        new THREE.Color(0xff0000).multiplyScalar(2), // Red x-axis
        new THREE.Color(0x00ff00).multiplyScalar(2), // Green y-axis 
        new THREE.Color(0x0000ff).multiplyScalar(2)  // Blue z-axis
    );
    axesHelper.position.set(-avgDimension / 20, -avgDimension / 20, - avgDimension / 20);
    scene.add(axesHelper);
}

function createRenderer() {
    const renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    return renderer;
}

function createCamera() {
    camera = new THREE.OrthographicCamera();
    camera.up.set(0, 0, 1);
    updateCamera();
    return camera;
}

function updateCamera() {
    const maxDimension = Math.max(dimensions.x, dimensions.y, dimensions.z) * (3 ** 0.5);
    const h = maxDimension * 1.2;
    const aspect = window.innerWidth / window.innerHeight;
    camera.left = -h * aspect / 2;
    camera.right = h * aspect / 2;
    camera.top = h / 2;
    camera.bottom = -h / 2;
    camera.position.set(-maxDimension / 3, -maxDimension * 2, dimensions.z);
    camera.updateProjectionMatrix();
}

function createControls() {
    controls = new OrbitControls(camera, renderer.domElement);
    // Set target to center of volume
    controls.target.set(dimensions.x / 2, dimensions.y / 2, dimensions.z / 2);
    return controls;
}

async function resetSimulation() {
    scene.remove(mesh);
    mesh = createVolumeMesh();
    controls.target.set(dimensions.x / 2, dimensions.y / 2, dimensions.z / 2);
    isPlaying = false;
    gui.controllers?.find((c) => c.property === "playPause")?.name(
        isPlaying ? "⏸ Pause" : "▶ Play"
    );

    updateCamera();
    updateAxesHelper();

    spheres.forEach(sphere => scene.remove(sphere));
    spheres = [];

    diffusionSim.cleanup();
    diffusionSim = await DiffusionSim.create(dimensions, volumeTexture.image.data as Float32Array, diffusionConstant);
}

function createGUI() {
    gui = new GUI();

    gui.add(dimensions, 'x', 8, 256, 16).onChange(() => {
        resetSimulation();
    });
    gui.add(dimensions, 'y', 8, 256, 16).onChange(() => {
        resetSimulation();
    });
    gui.add(dimensions, 'z', 8, 256, 16).onChange(() => {
        resetSimulation();
    });

    gui.add(
        { simIterationsPerFrame },
        'simIterationsPerFrame',
        1,
        100,
        1
    ).onChange((value: number) => {
        simIterationsPerFrame = value;
    }).name('Iterations Per Frame');

    gui.add(
        { diffusionConstant },
        'diffusionConstant',
        0.0,
        10.0,
        0.01
    ).onChange((value: number) => {
        diffusionConstant = value;
        diffusionSim.setDiffusionConstant(diffusionConstant);
    }).name('Diffusion Constant');

    gui.add(
        { renderstyle: 0 },
        'renderstyle',
        { 'Isosurface': 0, 'MIP': 1 }
    ).onChange((value: number) => {
        (mesh.material as THREE.ShaderMaterial).uniforms['u_renderstyle'].value = value;
    }).name('Render Style');



    gui.add({
        addSpheres: () => {
            addSpheresToScene(10, 5); // Default values: 10 spheres with radius 5
        }
    }, 'addSpheres').name('Add Spheres');

    gui.add({ reset: () => resetSimulation() }, 'reset')
        .name('Reset');

    gui.add({
        stepFrame: () => {
            if (!isPlaying) {
                stepSimulation();
            }
        }
    }, 'stepFrame').name('Step Frame');

    gui.add(
        {
            playPause: () => {
                isPlaying = !isPlaying;
                gui.controllers?.find((c) => c.property === "playPause")?.name(
                    isPlaying ? "⏸ Pause" : "▶ Play"
                );
            },
        },
        "playPause"
    ).name("▶ Play");
}

function generateVolumeData(pointSource: boolean = false) {
    const data = new Float32Array(dimensions.x * dimensions.y * dimensions.z);

    if (pointSource) {
        const centerX = Math.floor(dimensions.x / 2);
        const centerY = Math.floor(dimensions.y / 2);
        const centerZ = Math.floor(dimensions.z / 2);
        const centerIndex = centerX + centerY * dimensions.x + centerZ * dimensions.x * dimensions.y;
        data[centerIndex] = 1.0;
    } else {
        for (let i = 0; i < data.length; i++) {
            const x = i % dimensions.x;
            const y = Math.floor((i / dimensions.x) % dimensions.y);
            const z = Math.floor(i / (dimensions.x * dimensions.y));

            const dx = x - dimensions.x / 2;
            const dy = y - dimensions.y / 2;
            const dz = z - dimensions.z / 2;
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

            let val = Math.max(0, 1 - distance / (dimensions.x / 2)) + Math.random() * 0.3;

            if (val < 0.1) {
                val = 0.0;
            }

            data[i] = val;
        }
    }
    return data;
}

function createVolumeMesh() {
    // Create the 3D texture
    const data = generateVolumeData(true);
    volumeTexture = new THREE.Data3DTexture(
        data,
        dimensions.x,
        dimensions.y,
        dimensions.z
    );
    volumeTexture.format = THREE.RedFormat;
    volumeTexture.type = THREE.FloatType;
    volumeTexture.minFilter = volumeTexture.magFilter = THREE.LinearFilter;
    volumeTexture.unpackAlignment = 1;
    volumeTexture.needsUpdate = true;

    // Create colormap texture
    const colormapTexture = new THREE.TextureLoader().load('/cm_transparent_viridis.png');

    // Create shader material
    const uniforms = VolumeRenderShader1.uniforms;
    uniforms['u_data'].value = volumeTexture;
    uniforms['u_size'].value.set(dimensions.x, dimensions.y, dimensions.z);
    uniforms['u_clim'].value.set(0, 10);
    uniforms['u_renderstyle'].value = 0;
    uniforms['u_renderthreshold'].value = 0.5;
    uniforms['u_cmdata'].value = colormapTexture;

    const material = new THREE.ShaderMaterial({
        uniforms: uniforms,
        vertexShader: VolumeRenderShader1.vertexShader,
        fragmentShader: VolumeRenderShader1.fragmentShader,
        side: THREE.BackSide,
        transparent: true
    });

    // Create the volume mesh
    const geometry = new THREE.BoxGeometry(dimensions.x, dimensions.y, dimensions.z);
    geometry.translate(dimensions.x / 2 - 0.5, dimensions.y / 2 - 0.5, dimensions.z / 2 - 0.5);
    const mesh = new THREE.Mesh(geometry, material);

    // add an outline of the volume
    const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(dimensions.x, dimensions.y, dimensions.z));
    const outline = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
        color: 0x000000,
        depthTest: false // Ensure outline is always visible
    }));
    outline.position.set(dimensions.x / 2, dimensions.y / 2, dimensions.z / 2);
    outline.renderOrder = 1; // Render after other objects
    mesh.add(outline);

    scene.add(mesh);

    return mesh;
}

function updateVolumeMesh(newData: Float32Array) {
    volumeTexture.image.data = newData;
    volumeTexture.needsUpdate = true;

    (mesh.material as THREE.ShaderMaterial).uniforms['u_size'].value.set(dimensions.x, dimensions.y, dimensions.z);
    (mesh.material as THREE.ShaderMaterial).needsUpdate = true;
}

function addSpheresToScene(num_cells: number, radius: number) {
    // // Clear existing spheres if any
    // spheres.forEach(sphere => scene.remove(sphere));
    // spheres = [];

    // Create sphere geometry and material (shared between all spheres)
    const geometry = new THREE.SphereGeometry(radius, 32, 32);
    const material = new THREE.MeshBasicMaterial({
        color: 0x0000f0,
        transparent: true,
        opacity: 0.7
    });

    // Helper function to check if a position intersects with existing spheres
    function checkIntersection(x: number, y: number, z: number): boolean {
        for (const sphere of spheres) {
            const dx = sphere.position.x - x;
            const dy = sphere.position.y - y;
            const dz = sphere.position.z - z;
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (distance < radius * 2) { // 2 * radius is minimum distance between sphere centers
                return true;
            }
        }
        return false;
    }

    // Add spheres with collision detection
    let attempts = 0;
    const maxAttempts = 1000; // Prevent infinite loops

    for (let i = 0; i < num_cells && attempts < maxAttempts; i++) {
        let x, y, z;
        let intersects;

        // Keep trying new positions until we find one without intersection
        do {
            x = radius + Math.random() * (dimensions.x - 2 * radius);
            y = radius + Math.random() * (dimensions.y - 2 * radius);
            z = radius + Math.random() * (dimensions.z - 2 * radius);
            intersects = checkIntersection(x, y, z);
            attempts++;
        } while (intersects && attempts < maxAttempts);

        // Only add sphere if we found a valid position
        if (!intersects) {
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(x, y, z);
            scene.add(sphere);
            spheres.push(sphere);
        }
    }
}

function setupResizeHandler() {
    window.addEventListener('resize', () => {
        updateCamera();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}

async function animate() {
    stats.begin();
    controls.update();
    renderer.render(scene, camera);
    if (isPlaying) {
        stepSimulation();
    }
    stats.end();
}

async function stepSimulation() {
    console.log("stepping simulation");
    for (let i = 0; i < simIterationsPerFrame; i++) {
        await diffusionSim.process();
    }
    const volumeData = await diffusionSim.readResults();
    if (volumeData) {
        updateVolumeMesh(volumeData);
    }
}

export async function initThree() {
    stats = createStats();
    scene = createScene();
    renderer = createRenderer();
    camera = createCamera();
    controls = createControls();
    mesh = createVolumeMesh();
    diffusionSim = await DiffusionSim.create(dimensions, volumeTexture.image.data as Float32Array, diffusionConstant);
    createGUI();

    console.log(renderer.getContext());

    setupResizeHandler();
    renderer.setAnimationLoop(animate);
}