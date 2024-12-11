@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage> jitter: array<vec2f>; 
@group(0) @binding(2) var<storage> vbuffer: array<vec3f>; 
@group(0) @binding(3) var<storage> ibuffer: array<vec3u>; 
@group(0) @binding(4) var<storage> nbuffer: array<vec3f>;

const Pi = 3.1415926535; // Pi, but not
const lightPosition = vec3f(0.0, 1.0, 0.0);
const epsilon = 0.0001;

struct Uniforms {
    aspect: f32,
    camera_constant: f32,
    jitterSub: f32,
};

struct VSOut
{
    @builtin(position) position: vec4f,
    @location(0) coords : vec2f
};

struct Ray 
{
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32
};

struct Light
{
    L_i: vec3f,
    direction: vec3f,
    dist: f32
};

struct HitInfo
{
    has_hit: bool,
    dist: f32,
    position: vec3f,
    normal: vec3f,
    ambientColor: vec3f,
    diffuseColor: vec3f,
    shader: f32,
    ior1_over_ior2: f32,
    rho_s: f32,
    s: f32,
};

@fragment
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f
{
    const bgcolor = vec4f(0.1, 0.3, 0.6, 1.0);
    const max_depth = 10;
    var result = vec3f(0.0);
    for (var j = 0u; j < u32(uniforms.jitterSub*uniforms.jitterSub); j++) 
    {
        let uv = vec2f(coords.x*uniforms.aspect*0.5f, coords.y*0.5f);
        var r = get_camera_ray(uv+jitter[j]);
        var hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 0, 0, 0, 0);

        for(var i = 0; i < max_depth; i++) 
        {
            if(intersect_scene(&r, &hit)) { 
                result += shade(&r, &hit);
            }
            else { 
                result += bgcolor.rgb;
                break; 
            }

            if(hit.has_hit) { break; }
        }
    }
    result /= uniforms.jitterSub*uniforms.jitterSub;
    return vec4f(pow(result, vec3f(0.66)), bgcolor.a);
}

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex : u32) -> VSOut
{
    let unused = nbuffer[0];
    const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}



    
fn get_camera_ray(uv: vec2f) -> Ray
{
    const e = vec3f(0.15, 1.5, 10.0);
    const p = vec3f(0.15, 1.5, 0.0);
    const u = vec3f(0.0, 1.0, 0.0);
    var d = uniforms.camera_constant;
    var v = normalize(p-e);
    var b1 = normalize(cross(v, u));
    var b2 = cross(b1, v);
    var q = b1 * uv.x + b2 * uv.y + v * d;
    var omega = normalize(q);
    
    var ray = Ray(e, omega, 0.0, 1.0e32);
    return ray;
}


fn sample_point_light(pos: vec3f) -> Light
{
    const lightIntensity = vec3f(Pi, Pi, Pi); 
    
    var L_i = lightIntensity / dot((lightPosition - pos), (lightPosition - pos));
    var l = lightPosition - pos;
    var dist = length(l);
    
    var light = Light(L_i, normalize(l), dist);
    return light;
}

fn sample_directional_light() -> Light 
{
    const intensityGAINS = 1;
    const lightIntensity = vec3f(Pi*intensityGAINS, Pi*intensityGAINS, Pi*intensityGAINS);
    var L_i = lightIntensity;
    var l = normalize(vec3f(-1.0, -1.0, -1.0));
    var dist = 1e32;
    var light = Light(L_i, l, dist);
    return light;
}



fn intersect_scene(r: ptr<function, Ray>, hit : ptr<function, HitInfo>) -> bool
{
    var triangleRGB = vec3f(0.9);
    (*r).tmin = epsilon;
    for (var i = 0u; i < arrayLength(&ibuffer); i++) 
    {
        if (intersect_triangle(*r, hit, i)) {
            (*r).tmax = min((*hit).dist, (*r).tmax);
            (*hit).has_hit = true;
            if ((*r).tmax == (*hit).dist) {
                (*hit).diffuseColor = triangleRGB;
            }
        }
    }
    return (*hit).has_hit;
}

fn intersect_triangle(r: Ray, hit: ptr<function, HitInfo>, index: u32) -> bool 
{
    let temp = ibuffer[index]; 
    var e0 = vbuffer[temp.y] - vbuffer[temp.x];
    var e1 = vbuffer[temp.z] - vbuffer[temp.x];
    var n = cross(e0, e1);
    
    var t = (dot((vbuffer[temp.x] - r.origin), n)) / dot(r.direction, n);
    
    var objectHit = false;
    if ((t < r.tmax) & (t > r.tmin)) {
        var beta = dot(cross((vbuffer[temp.x]-r.origin), r.direction), e1)/dot(r.direction, n);
        var gamma = -dot(cross((vbuffer[temp.x]-r.origin), r.direction), e0)/dot(r.direction, n);
        var alpha = 1 - beta - gamma;
        objectHit = (beta >= 0) & (gamma >= 0) & ((beta + gamma) <= 1);
        if (objectHit) {
            if ((*hit).dist > 0) {
                (*hit).dist = min((*hit).dist, t);
            }
            else {
                (*hit).dist = t;
            }
            (*hit).position = r.origin + t * r.direction;
            let x = alpha * nbuffer[temp.x] + beta * nbuffer[temp.y] + gamma * nbuffer[temp.z];
            (*hit).normal = normalize(x);
        }
    }
    
    return objectHit;
    
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f
{   
    var light = sample_directional_light();
    var ray = Ray((*hit).position, -light.direction, 0.0, light.dist);
    var temp = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 0, 0, 0, 0);
    return ((*hit).diffuseColor/Pi) * 1 * light.L_i * dot(-light.direction, (*hit).normal);
}


    