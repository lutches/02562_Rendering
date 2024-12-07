"use strict";

window.onload = function () { main(); }



async function main() {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const canvas = document.getElementById("webgpu-canvas");
    const context = canvas.getContext("gpupresent") || canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: device, format: canvasFormat, });


    const wgsl = device.createShaderModule({
        code: await (await fetch("./p2.wgsl")).text()
    });
    
    var sphereMenu = document.getElementById("sphereMenu");
    sphereMenu.addEventListener("change", function (ev) {
        uniforms[2] = sphereMenu.value;
        animate();
    });

    var materialMenu = document.getElementById("materialMenu");
    materialMenu.addEventListener("change", function (ev) {
        uniforms[3] = materialMenu.value;
        animate();
    });

    var imageStyle = document.getElementById("imageStyle");
    imageStyle.addEventListener("change", function (ev) {
        uniforms_ui[0] = imageStyle.value;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        animate();

    });

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const aspect = canvas.width / canvas.height;
    var cam_const = 1.0;
    const sphereMaterial = 5.0;
    const material = 1;
    var uniforms = new Float32Array([aspect, cam_const, sphereMaterial, material]);
    var uniforms_ui = new Uint32Array([1, 0]);



    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex:
        {
            module: wgsl,
            entryPoint: "main_vs",
        },
        fragment: {
            module: wgsl,
            entryPoint: "main_fs",
            targets: [{ format: canvasFormat }]
        },
        primitive: {
            topology: "triangle-strip",
        },
    });
    
    
    
    const uniformBuffer = device.createBuffer({
        size: uniforms.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const my_sampler = device.createSampler({
        addressModeU: "repeat",
        addressModeV: "repeat",
        minFilter: "nearest",
        magFilter: "nearest",
    });

    device.queue.writeBuffer(uniformBuffer, 0, uniforms);

    const texture = await load_texture(device, "../textures/grass.jpg");

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: my_sampler },
            { binding: 2, resource: texture.createView() }
        ],
    });

    addEventListener("keydown", (event) => {
        if (event.key === "ArrowUp") {
            cam_const *= 1.5;
            requestAnimationFrame(animate);
        } else if (event.key === "ArrowDown") {
            cam_const /= 1.5;
            requestAnimationFrame(animate);
        }
    });

    addEventListener("wheel", function (ev) {
        let zoom = ev.deltaY > 0 ? 0.95 : 1.05;
        cam_const *= zoom;
        animate()
    });


    function animate() {
        uniforms[1] = cam_const;
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        render(device, context, pipeline, bindGroup);
    }

    animate();

    function render(device, context, pipeline, bindGroup) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass(
            {
                colorAttachments:
                    [{
                        view: context.getCurrentTexture().createView(),
                        loadOp: "clear",
                        storeOp: "store",
                    }]
            });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(4);
        pass.end();
        device.queue.submit([encoder.finish()])
    }
}


async function load_texture(device, filename) {
    const response = await fetch(filename);
    const blob = await response.blob();
    const img = await createImageBitmap(blob, { colorSpaceConversion: 'none' });

    const texture = device.createTexture({
        size: [img.width, img.height, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    });

    device.queue.copyExternalImageToTexture(
        { source: img, flipY: true },
        { texture: texture },
        { width: img.width, height: img.height },
    );
    return texture;
}