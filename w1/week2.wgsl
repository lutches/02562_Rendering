struct Uniforms {
    aspect: f32,
    cam_const: f32,
    gamma: f32
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords: vec2f,
};
@vertex
fn main_vs(@builtin(vertex_index) VertexIndex: u32) -> VSOut {
    const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}


// Define Ray struct

struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32
};
// Create a struct HitInfo 
struct HitInfo {
    has_hit: bool,
    dist: f32,
    position: vec3f,
    normal: vec3f,
    color: vec3f,
    shader: u32
};

struct Light {
    L_i: vec3f,
    w_i: vec3f,
    dist: f32
};

fn intersect_scene(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    // Define scene data as constants.
    if (intersect_plane(*r, hit, vec3f(0), vec3f(0,1,0), vec4f(0.1,0.7,0.0,1.0))){r.tmax = (*hit).dist;}
    if (intersect_triangle(*r, hit, array<vec3f, 3>(vec3f(-0.2,0.1,0.9), vec3f(0.2,0.1,0.9), vec3f(-0.2, 0.1, -0.1)), vec4f(0.4,0.3,0.2,1))){r.tmax = (*hit).dist;}
    if (intersect_sphere(*r, hit, vec3f(0,0.5,0), 0.3, vec4f(0,0,0,1))){r.tmax = (*hit).dist;}
    // Call an intersection function for each object.
    // For each intersection found, update (*r).tmax and store additional info about the hit.
    return (*hit).has_hit;
}

fn intersect_plane(r: Ray, hit: ptr<function, HitInfo>, p: vec3f, n: vec3f, c: vec4f) -> bool 
{
let denom = dot(r.direction, n);
if (abs(denom) > 0) {
    let t = dot(p - r.origin, n) / denom;
    if (t >= r.tmin && t <= r.tmax) {
        *hit = HitInfo(true, t, r.origin + t * r.direction, n, c.rgb,1);
        return true;
    }
}
return false;
}
fn intersect_triangle(r: Ray, hit: ptr<function, HitInfo>, v: array<vec3f,3>, c: vec4f) -> bool 
{

    let e0 = v[0]-v[1];
    let e1 = v[0]-v[2];
    let n = cross(e0,e1 );
    let denom = dot(r.direction, n);
    if (denom != 0){
        let t = dot(v[0]-r.origin, n)/ denom;
        if (t>= r.tmin && t <= r.tmax) {
            let beta = dot(cross((v[0]-r.origin), r.direction), e1)/denom;
            let gamma = dot(cross((v[0]-r.origin), r.direction), e0)/denom;
            if(beta >= 0 && gamma >= 0 && beta + gamma <= 1){
                *hit = HitInfo(true, t, r.origin + t * r.direction, n, c.rgb,1);
                return true;
            }
        }
    }
    return false;
}

fn intersect_sphere(r: Ray, hit: ptr<function, HitInfo>, position: vec3f, radius: f32, color: vec4f) -> bool 
{
    let bd2 = dot((r.origin - position), r.direction); 
    let c = dot((r.origin - position), (r.origin - position)) - radius * radius;
    if (bd2*bd2 - c >= 0){
        let t1 = -bd2 - sqrt(bd2 * bd2 - c);
        if (t1 >= r.tmin && t1 <= r.tmax){
            let hitPoint = r.origin + t1 * r.direction;
            let normal = hitPoint - position;
            *hit = HitInfo(true, t1,hitPoint, normal, color.rgb,1);
            return true;
        }
        let t2 = -bd2 + sqrt(bd2 * bd2 - c);
        if(t2 >= r.tmin && t2 <= r.tmax){
            let hitPoint2 = r.origin + t2 * r.direction;
            let normal2 = hitPoint2 - position;
            *hit = HitInfo(true, t2 , hitPoint2, normal2, color.rgb,1);
            return true;
        }
        
    }
    return false;
}

fn lambertian(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f
{
    let light = sample_point_light((*hit).position);
    let L = normalize(light.w_i);
    let N = normalize((*hit).normal);
    let diffuse = max(dot(N, L), 0.0) * (*hit).color;
    return light.L_i * diffuse;
}
//fn phong(r: Ray, hit: ptr<function, HitInfo>, position: vec3f, normal: vec3f) -> Ray {}
//fn mirror(r: Ray, hit: ptr<function, HitInfo>, position: vec3f, normal: vec3f) -> Ray {}

fn sample_point_light(pos: vec3f) -> Light
{
    const l_pos = vec3f(0,1,0);
    let L_i = vec3f(1);
    let w_i = normalize(l_pos - pos);
    let dist = length(l_pos - pos);
    return Light(L_i, w_i, dist);
};

fn shade(r: ptr<function, Ray>, hit: ptr<function,HitInfo>) -> vec3f
{
    switch (*hit).shader {
        case 1 {return lambertian(r,hit);}
        case default {return (*hit).color;}
    }
}
fn get_camera_ray(ipcoords: vec2f) -> Ray {
    // Implement ray generation

    const eye = vec3f(2.0, 1.5, 2.0);
    const p = vec3f(0.0, 0.5, 0.0);
    const up = vec3f(0.0, 1.0, 0.0);
    const d = 1.0;
    
    const v = normalize(p-eye);
    const b1 = normalize(cross(v,up));
    const b2 = cross(b1,v);
    const origin = eye;
    let direction = normalize(ipcoords.x*b1+ipcoords.y*b2+v*uniforms.cam_const);
    
    const tmin = 0.0;
    const tmax = 100.0;
    
    return Ray(origin, direction, tmin, tmax);
}
@fragment
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f {
    const bgcolor = vec4f(0.1, 0.3, 0.6, 1.0);
    const max_depth = 10;
    let uv = vec2f(coords.x * uniforms.aspect * 0.5f, coords.y * 0.5f);
    var r = get_camera_ray(uv);
    var result = vec3f(0.0);
    var hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0),0);
    
    for (var i = 0; i < max_depth; i++) {
        if (intersect_scene(&r, &hit)) {
            result += shade(&r, &hit);
        } else {
            result += bgcolor.rgb;
            break;
        }
        if (hit.has_hit) {
            break;
        }
    }
    
    return vec4f(pow(result, vec3f(1.0 / uniforms.gamma)), bgcolor.a);
} 
    