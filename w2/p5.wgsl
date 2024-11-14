struct Uniforms {
    aspect: f32,
    cam_const: f32,
    sphere_material: f32,
    material: f32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords: vec2f,
};

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex: u32) -> VSOut 
{
    const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32
};

struct HitInfo {
    has_hit: bool,
    dist: f32,
    position: vec3f,
    normal: vec3f,
    diffuseColor: vec3f,
    ambientColor: vec3f,
    shader: f32,
    IoR: f32
};

struct Light {
    L_i: vec3f,
    w_i: vec3f,
    dist: f32
};

const l_pos = vec3f(0,1,0);
const sphereIR = 1.5;
const sphereShininess = 42;
const specualarReflectance = 0.1;
const Pi = 3.1415;

fn sample_point_light(pos: vec3f) -> Light
{
    var l = l_pos - pos;
    var L_i = vec3f(Pi)/ dot(l, l);

    let w_i = normalize(l);
    let dist = length(l);
    return Light(L_i, w_i, dist);
};

fn intersect_scene(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    // Define scene data as constants.

    // Plane
    const planePosition = vec3f(0);
    const planeNormal = vec3f(0,1,0);
    const planeRGB = vec3f(0.1,0.7,0.0);

    // Triangle
    const v0 = vec3f(-0.2,0.1,0.9);
    const v1 = vec3f(0.2,0.1,0.9);
    const v2 = vec3f(-0.2, 0.1, -0.1);
    const triangleRGB = vec3f(0.4,0.3,0.2);

    // Ball
    const sphereposition = vec3f(0.0,0.5,0.0);
    const sphereRadius = 0.3;
    const sphereRGB = vec3f(0,0,0);
    
    // Call an intersection function for each object.
    if (intersect_plane(*r, hit, planePosition, planeNormal))
    {
        (*r).tmax = min((*hit).dist, (*r).tmax);
        (*hit).has_hit = true;
        (*hit).shader = uniforms.material;
        if ((*r).tmax == (*hit).dist) {
            (*hit).diffuseColor = planeRGB*0.9;
            (*hit).ambientColor = planeRGB*0.1;
        }
    }
    if (intersect_triangle(*r, hit, array<vec3f,3>(v0, v1, v2)))
    {
        (*r).tmax = min((*hit).dist, (*r).tmax);
        (*hit).has_hit = true;
        (*hit).shader = uniforms.material;
        if ((*r).tmax == (*hit).dist) 
        {
            (*hit).normal = normalize(cross(v1 - v0, v2 - v0));
            (*hit).diffuseColor = triangleRGB*0.9;
            (*hit).ambientColor = triangleRGB*0.1;
        }
    }
    if (intersect_sphere(*r, hit, sphereposition, sphereRadius)) {
            (*r).tmax = min((*hit).dist, (*r).tmax);
            (*hit).has_hit = true;
            (*hit).shader = uniforms.sphere_material;
            if ((*r).tmax == (*hit).dist) {
                (*hit).diffuseColor = sphereRGB*0.9;
                (*hit).ambientColor = sphereRGB*0.1;
            }
        }

    // For each intersection found, update (*r).tmax and store additional info about the hit.
    return (*hit).has_hit;
}

fn intersect_plane(r: Ray, hit: ptr<function, HitInfo>, p: vec3f, n: vec3f) -> bool 
{
    let denom = dot(r.direction, n);
    var t = dot(p - r.origin, n) / denom;

    var objectHit = (t< r.tmax) & (t > r.tmin);

    if (objectHit) {
        if (abs(denom) > 0) {
            if ((*hit).dist > 0)
            {
                (*hit).dist = min((*hit).dist, t);
            }
            else {(*hit).dist = t;}
            (*hit).position = r.origin + t * r.direction;
            (*hit).normal = n;
        }
    }
    return objectHit;
}

fn intersect_triangle(r: Ray, hit: ptr<function, HitInfo>, v: array<vec3f,3>) -> bool 
{
    let e0 = v[0]-v[1];
    let e1 = v[0]-v[2];
    let n = cross(e0,e1 );

    var denom = dot(r.direction, n);
    var t = dot(v[0]-r.origin, n)/ denom;

    var objectHit = false;
    if ((t < r.tmax) && (t > r.tmin))
    {
        let beta = dot(cross((v[0]-r.origin), r.direction), e1)/denom;
        let gamma = dot(cross((v[0]-r.origin), r.direction), e0)/denom;
        objectHit = (beta >= 0) & (gamma >= 0) & ((beta + gamma) <= 1);
    }
    if (objectHit) {
            if ((*hit).dist > 0) {
                (*hit).dist = min((*hit).dist, t);
            }
            else {
                (*hit).dist = t;
            }
            (*hit).position = r.origin + t * r.direction;
            (*hit).normal = n;
        }
    else {
        objectHit = false;
    }
    return objectHit;
}

fn intersect_sphere(r: Ray, hit: ptr<function, HitInfo>, position: vec3f, radius: f32) -> bool 
{
    var e = r.origin;
    var d = r.direction;
    var disc = (dot(d, (e - position)) * dot(d, (e - position))) - dot(d, d) * (dot((e - position), (e - position)) - (radius*radius));
    var objectHit = (disc >= 0);
    var t1 = dot(-d, (e - position)) - sqrt(disc) / dot(d, d);
    var t2 = dot(-d, (e - position)) + sqrt(disc) / dot(d, d);
    if (t1 < r.tmax && t1 > r.tmin) {
        if ((*hit).dist > 0) {
            (*hit).dist = min((*hit).dist, t1);
        }
        else {
            (*hit).dist = t1;
        }
        (*hit).position = r.origin + t1 * r.direction;
        (*hit).normal = normalize((*hit).position - position);
    }
    else if (t2 < r.tmax && t2 > r.tmin) {
        if ((*hit).dist > 0) {
            (*hit).dist = min((*hit).dist, t2);
        }
        else {
            (*hit).dist = t2;
        }
        (*hit).position = r.origin + t2 * r.direction;
        (*hit).normal = normalize((*hit).position - position);
    }
    else {
        objectHit = false;
    }
    
    return objectHit;
}

fn lambertian(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f
{
    var light = sample_point_light((*hit).position);
    var ray = Ray(l_pos, -light.w_i, 0.0, light.dist-0.01);
    var temp = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 0, 1);

    if (intersect_scene(&ray, &temp)) 
    {
        return vec3f(0.0);
    }
    else 
    {
        return ((*hit).diffuseColor/Pi) * light.L_i * (max(0, dot((*hit).normal, light.w_i)));
    }
}

fn mirror(r: ptr<function, Ray>, hit: ptr<function,HitInfo>) -> vec3f
{
    (*r).direction = 2*(-dot((*r).direction, (*hit).normal))*(*hit).normal + (*r).direction;
    (*r).origin = (*hit).position;
    (*r).tmin = 0.0001;

    (*hit).has_hit = false;
    return vec3f(0,0,0);
}

fn refrection(r:ptr<function, Ray>, hit: ptr<function,HitInfo>) -> vec3f
{
    if (dot((*r).direction, (*hit).normal) < 0) {
        (*hit).IoR = 1.0/sphereIR;
    }
    else {
        (*hit).IoR = sphereIR;
        (*hit).normal = -(*hit).normal;
    }
    
    var cosSqTheta = 1 - (*hit).IoR * (*hit).IoR * (1 - dot((*r).direction, (*hit).normal) * dot((*r).direction, (*hit).normal));

    if (cosSqTheta < 0) {
        return mirror(r,hit);
    }
    else {
        (*r).direction = (*hit).IoR * (dot(-(*r).direction, (*hit).normal) * (*hit).normal + (*r).direction) - (*hit).normal * sqrt(cosSqTheta);
        (*hit).has_hit = false;
        (*r).origin = (*hit).position;
        (*r).tmin = 0.0001;
        (*r).tmax = 1.0e32;
        return vec3f(0,0,0);
    }
}

fn phong(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f
{
    var light = sample_point_light((*hit).position);
    var Lr = ((*hit).diffuseColor/Pi + specualarReflectance * (sphereShininess + 2)/
    (2*Pi) * pow(max(0, dot(-(*r).direction, reflect(-light.w_i, (*hit).normal))), sphereShininess)) *
    light.L_i * max(0, dot(light.w_i, (*hit).normal));

    return Lr;
}

fn shade(r: ptr<function, Ray>, hit: ptr<function,HitInfo>) -> vec3f
{
    switch i32((*hit).shader) {
        case 1 {return (lambertian(r,hit) + (*hit).ambientColor);}
        case 2 {return mirror(r,hit);}
        case 3 {return refrection(r,hit);}
        case 4 {return phong(r,hit);}
        case 5{ return phong(r,hit) + refrection(r,hit);}
        case default { return (*hit).ambientColor + (*hit).diffuseColor; }
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
    
    return Ray(origin, direction, 0.0, 1.0e32);
}
@fragment
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f {
    const bgcolor = vec4f(0.1, 0.3, 0.6, 1.0);
    const max_depth = 10;
    let uv = vec2f(coords.x * uniforms.aspect * 0.5f, coords.y * 0.5f);
    var r = get_camera_ray(uv);
    var result = vec3f(0.0);
    var hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0),0, 1);
    
    for (var i = 0; i < max_depth; i++) {
        if (intersect_scene(&r, &hit)) {
            result += shade(&r, &hit);
        } 
        else {
            result += bgcolor.rgb;
            break;
        }
        if (hit.has_hit) {break;}
    }
    
    return vec4f(pow(result, vec3f(1.0)), bgcolor.a);
} 
    