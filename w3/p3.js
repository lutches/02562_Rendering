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
        code: await (await fetch("./p3.wgsl")).text()
    });


    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const aspect = canvas.width / canvas.height;
    var cam_const = 1.0;
    const sphereMaterial = 5.0;
    const material = 1;
    const jitterSub = 4;
    const scalingFactor = 0.2;
    var groupNumber = 1;
    var uniforms = new Float32Array([aspect, cam_const, sphereMaterial, material, jitterSub, scalingFactor]);

    let jitter = new Float32Array(200); // allowing subdivs from 1 to 10  
    const jitterBuffer = device.createBuffer(
        {
            size: jitter.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
        });



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

    const texture = await load_texture(device, "../textures/grass.jpg");
    texture.samplers = [];

    texture.samplers.push(
        device.createSampler({
            addressModeU: "clamp-to-edge",
            addressModeV: "clamp-to-edge",
            minFilter: "linear",
            magFilter: "linear",
        }));

    texture.samplers.push(
        device.createSampler({
            addressModeU: "repeat",
            addressModeV: "repeat",
            minFilter: "nearest",
            magFilter: "nearest",
        }));

    device.queue.writeBuffer(uniformBuffer, 0, uniforms);

    var bindGroups = [];
    for (var i = 0; i < 2; i++) {
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: texture.samplers[i] },
                { binding: 2, resource: texture.createView() },
                { binding: 3, resource: { buffer: jitterBuffer } },
            ],
        });
        bindGroups.push(bindGroup);
    }

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

    var increase = document.getElementById("increase");
    increase.addEventListener("click", function (ev) {
        if (uniforms[4] < 10) {
            uniforms[4]++;
        }

        requestAnimationFrame(animate);
    });

    var decrease = document.getElementById("decrease");
    decrease.addEventListener("click", function (ev) {
        if (uniforms[4] > 1) {
            uniforms[4]--;
        }
        console.log(uniforms[4]);
        requestAnimationFrame(animate);
    });

    var imageStyle = document.getElementById("imageStyle");
    imageStyle.addEventListener("change", function (ev) {
        groupNumber = parseInt(imageStyle.value);
        console.log(groupNumber);
        animate();
    });

    window.addEventListener("resize", function () {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        uniforms[0] = canvas.width / canvas.height;
        animate();
    });

    function animate() {
        uniforms[1] = cam_const;
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        render(device, context, pipeline);
    }

    animate();

    function render(device, context, pipeline) {
        compute_jitters(jitter, 1 / canvas.height, uniforms[4]);
        device.queue.writeBuffer(jitterBuffer, 0, jitter);
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
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
        pass.setBindGroup(0, bindGroups[groupNumber]);
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


function compute_jitters(jitter, pixelsize, subdivs) {
    const step = pixelsize / subdivs;
    if (subdivs < 2) {
        jitter[0] = 0.0; jitter[1] = 0.0;

    }
    else {
        for (var i = 0; i < subdivs; ++i)
            for (var j = 0; j < subdivs; ++j) {
                const idx = (i * subdivs + j) * 2; jitter[idx] = (Math.random() + j) * step - pixelsize * 0.5;
                jitter[idx + 1] = (Math.random() + i) * step - pixelsize * 0.5;
            }
    }
}