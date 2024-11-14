struct Uniforms_f {
    aspect: f32,
};

struct Uniforms_ui {
    use_repeat: u32,
    use_linear: u32,
};

struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords: vec2f,
};

@group(0) @binding(0) var<uniform> uniforms_f: Uniforms_f;
@group(0) @binding(1) var<uniform> uniforms_ui: Uniforms_ui;
@group(0) @binding(2) var my_texture: texture_2d<f32>;

fn texture_nearest(texture: texture_2d<f32>, texcoords: vec2f, repeat: bool) -> vec3f 
{       
    let res = textureDimensions(texture);
    let st = select(clamp(texcoords, vec2f(0), vec2f(1)), texcoords - floor(texcoords), repeat);
    let ab = st*vec2f(res);       
    let UV = vec2u(ab + 0.5) % res;      
    let texcolor = textureLoad(texture, UV, 0);      
    return texcolor.rgb;     
}    
fn texture_linear(texture: texture_2d<f32>, texcoords: vec2f, repeat: bool) -> vec3f 
{       
    let res = textureDimensions(texture);
    let st = select(clamp(texcoords, vec2f(0), vec2f(1)), texcoords - floor(texcoords), repeat);
    let ab = st * vec2f(res) - 0.5;
    let iuv = vec2i(floor(ab));
    let fuv = fract(ab);
    
    let uv00 = (iuv + vec2i(0, 0)) % vec2i(res);
    let uv10 = (iuv + vec2i(1, 0)) % vec2i(res);
    let uv01 = (iuv + vec2i(0, 1)) % vec2i(res);
    let uv11 = (iuv + vec2i(1, 1)) % vec2i(res);
    
    let c00 = textureLoad(texture, uv00, 0).rgb;
    let c10 = textureLoad(texture, uv10, 0).rgb;
    let c01 = textureLoad(texture, uv01, 0).rgb;
    let c11 = textureLoad(texture, uv11, 0).rgb;
    
    let texcolor = mix(mix(c00, c10, fuv.x), mix(c01, c11, fuv.x), fuv.y);

    return texcolor.rgb;     
}

@fragment

fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f 
{
    let uv = vec2f(coords.x * uniforms_f.aspect * 0.5, coords.y * 0.5);
    let use_repeat = uniforms_ui.use_repeat != 0;
    let use_linear = uniforms_ui.use_linear != 0;
    let color = select(
        texture_nearest(my_texture, uv, use_repeat),
        texture_linear(my_texture, uv, use_repeat),
        use_linear
    );
    return vec4f(color, 1.0);
}

 @vertex
    fn main_vs(@builtin(vertex_index) VertexIndex : u32) -> VSOut
    {
        const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
        var vsOut: VSOut;
        vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
        vsOut.coords = pos[VertexIndex];
        return vsOut;
    }


