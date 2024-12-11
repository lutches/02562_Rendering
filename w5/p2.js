"use strict";

// Entry point for the application
window.onload = function () {
    main();
};

// Main application function
async function main() {
    const gpu = navigator.gpu;
    const adapter = await gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const canvas = document.getElementById("webgpu-canvas");
    const context = canvas.getContext("gpupresent") || canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    configureCanvasContext(context, device, canvasFormat);

    const pixelSize = 1 / canvas.height;
    const objFilename = '../objects/teapot.obj';
    const drawingInfo = await loadOBJFile(objFilename, 1, true);

    const vBuffer = createBuffer(device, drawingInfo.vertices, GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE);
    const iBuffer = createBuffer(device, drawingInfo.indices, GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE);

    const {uniforms, uniformBuffer} = createUniformBuffer(device, canvas);
    const jitterBuffer = createJitterBuffer(device, 200);

    const pipeline = await createRenderPipeline(device, canvasFormat);

    // Setup event listeners
    setupEventListeners(uniforms, uniformBuffer, device, animate);

    let bindGroup;

    function animate() {
        bindGroup = createBindGroup(device, pipeline, uniformBuffer, jitterBuffer, vBuffer, iBuffer);
        render();
    }

    function render() {
        computeJitters(jitterBuffer, device, pixelSize, uniforms[2]);
        executeRenderPass(device, context, pipeline, bindGroup);
    }

    animate();
}

// Configure the canvas context
function configureCanvasContext(context, device, format) {
    context.configure({
        device: device,
        format: format,
    });
}

// Load the OBJ file
async function loadOBJFile(filename, scale, ccw) {
    return await readOBJFile(filename, scale, ccw);
}

// Create a GPU buffer
function createBuffer(device, data, usage) {
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage: usage,
    });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
}

// Create the uniform buffer
function createUniformBuffer(device, canvas) {
    const aspect = canvas.width / canvas.height;
    const cameraConstant = 2.5;
    const jitterSub = 1;
    
    const uniforms = new Float32Array([aspect, cameraConstant, jitterSub, 0]);

    const uniformBuffer = device.createBuffer({
        size: uniforms.byteLength, // 16 bytes for 4 floats
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniforms);

    return {uniforms, uniformBuffer};
}

// Create the jitter buffer
function createJitterBuffer(device, length) {
    const jitter = new Float32Array(length);
    const buffer = device.createBuffer({
        size: jitter.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    return buffer;
}

// Create the render pipeline
async function createRenderPipeline(device, format) {
    const shaderModule = device.createShaderModule({
        code: await (await fetch("./p2.wgsl")).text()
    });

    return device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: shaderModule,
            entryPoint: "main_vs",
        },
        fragment: {
            module: shaderModule,
            entryPoint: "main_fs",
            targets: [{ format: format }],
        },
        primitive: {
            topology: "triangle-strip",
        },
    });
}

// Set up event listeners
function setupEventListeners(uniforms, uniformBuffer, device, animatecallback) {
    addEventListener("wheel", function (ev) {
        const zoom = ev.deltaY > 0 ? 0.95 : 1.05;
        uniforms[1] *= zoom;
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        animatecallback();
    });
}

// Create a bind group
function createBindGroup(device, pipeline, uniformBuffer, jitterBuffer, vBuffer, iBuffer) {
    return device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: jitterBuffer } },
            { binding: 2, resource: { buffer: vBuffer } },
            { binding: 3, resource: { buffer: iBuffer } },
        ],
    });
}

// Compute jitters
function computeJitters(jitterBuffer, device, pixelSize, jitterSub) {
    const jitter = new Float32Array(jitterBuffer.size / Float32Array.BYTES_PER_ELEMENT);
    const step = pixelSize / jitterSub;

    if (jitterSub < 2) {
        jitter[0] = 0.0;
        jitter[1] = 0.0;
    } else {
        for (let i = 0; i < jitterSub; ++i) {
            for (let j = 0; j < jitterSub; ++j) {
                const idx = (i * jitterSub + j) * 2;
                jitter[idx] = (Math.random() + j) * step - pixelSize * 0.5;
                jitter[idx + 1] = (Math.random() + i) * step - pixelSize * 0.5;
            }
        }
    }

    device.queue.writeBuffer(jitterBuffer, 0, jitter);
}

// Execute the render pass
function executeRenderPass(device, context, pipeline, bindGroup) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            storeOp: "store",
        }],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(4);
    pass.end();

    device.queue.submit([encoder.finish()]);
}
