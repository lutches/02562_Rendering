struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) coords: vec2f
}

struct UniformsF { // float values
  aspect: f32,
  cam_const: f32,
}
struct UniformsInt { // unsigned int values
  glass_shader: u32,
  matte_shader: u32,
  texture_enabled: u32,
  subdivisions: u32, // sqrt of number of subpixels
  obj_idx: u32,
}

// Axis-Aligned Bounding box (box between the points min and max)
struct AABB {
  min: vec3f,
  max: vec3f,
};

struct VertexAttrib {
  pos: vec4f,
  normal: vec4f,
}

struct Material {
  diffuse: vec4f,
  emission: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms_f: UniformsF;
@group(0) @binding(1) var<uniform> uniforms_int: UniformsInt;
// 3x3 jitter, 2 flaots (x, y) per point
// rounded up to a multiple of 16 bytes
@group(0) @binding(2) var<uniform> subpixels: array<vec4f, 9>; 
@group(0) @binding(3) var<uniform> aabb: AABB;

@group(0) @binding(4) var<storage> vert_attribs: array<VertexAttrib>;
@group(0) @binding(6) var<storage> materials: array<Material>; // colors
@group(0) @binding(7) var<storage> vert_indices: array<vec4u>;
@group(0) @binding(8) var<storage> treeIds: array<u32>;
@group(0) @binding(9) var<storage> bspTree: array<vec4u>;
@group(0) @binding(10) var<storage> bspPlanes: array<f32>;

@group(0) @binding(11) var<storage> light_indices: array<u32>;


@vertex
fn main_vs(@builtin(vertex_index) VertexIndex: u32) -> VSOut {
  const pos = array<vec2f, 4>(vec2f(-1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(1, -1));
  var vsOut: VSOut;
  vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
  vsOut.coords = pos[VertexIndex];
  return vsOut;
}

struct Ray {
  origin: vec3f,
  direction: vec3f,
  tmin: f32,
  tmax: f32,
}

struct HitInfo {
  hit: bool,
  distance: f32,
  position: vec3f,
  normal: vec3f,
  color_amb: vec3f, // ambient component of color (always on)
  color_diff: vec3f, // diffuse (lambertian) component of color
  color_specular: vec3f, // specular reflectance
  shine: f32, // shininess Phong exponent
  refractive_ratio: f32, // ratio of incident to transmitted refractive index
  shader: u32,
  texcoords: vec2f, // xy coordinates of texture collision point
}
// shader type names:
const shader_reflect: u32 = 2;
const shader_refract: u32 = 3;
const shader_phong: u32 = 4;
const shader_glossy: u32 = 5;
const shader_matte: u32 = 1;


const tex_scale = 0.02;
// these are all arrays so we can choose the settings
// for the loaded OBJ file
// eye point
// these are in the order bunny, teapot, dragon
const e = vec3f(277.0, 275.0, -570.0);
// camera constant
const d = 1.0f;
// up vector
const u = vec3f(0, 1, 0);
// look at point
const p = vec3f(277.0, 275.0, 0.0);
// directional light direction
const dir_light_dir = vec3f(-0.3, -0.1, -0.8);

struct Onb { // orthonormal basis for plane
  tangent: vec3f,
  binormal: vec3f,
  normal: vec3f,
};


fn get_camera_ray(ipcoords: vec2f) -> Ray {
  let v = normalize(p - e);

  let b1 = normalize(cross(v, u));
  let b2 = cross(b1, v); // b1 and v are magnitude 1 so their cross is already 1
  
  let x = ipcoords[0];
  let y = ipcoords[1];

  let q = (v)*d + (b1 * x) + (b2 * y);

  var dir = normalize(q);
  return Ray(e, dir, 0.1, 10000);
}

fn int_scene(ray: ptr<function, Ray>,  hit: ptr<function, HitInfo>) -> bool {
  const sphere_c = vec3f(0.0, 0.5, 0.0);
  const sphere_r = 0.3;

  const plane_point = vec3f(0.0, 0.0, 0.0);
  const plane_onb = Onb(
    vec3f(-1.0, 0.0, 0.0),
    vec3f(0.0, 0.0, 1.0),
    vec3f(0.0, 1.0, 0.0)
  );
  // check outer binding box
  if(int_aabb(ray)){
    if(int_trimesh(ray, hit)){
      (*ray).tmax = (*hit).distance;
    }
  }

  return (*hit).hit;
}

fn int_aabb(r: ptr<function, Ray>) -> bool {
  let p1 = (aabb.min - r.origin) / r.direction;
  let p2 = (aabb.max - r.origin) / r.direction;

  let pmin = min(p1, p2);
  let pmax = max(p1, p2);
  let tmin = max(pmin.x, max(pmin.y, pmin.z));
  let tmax = min(pmax.x, min(pmax.y, pmax.z));
  if(tmin > tmax || tmin > r.tmax || tmax < r.tmin){
    // ray doesnt intersect AABB
    return false;
  }
  // ray does intersect, constrain search to AABB
  r.tmin = max(tmin - 1.0e-3f, r.tmin);
  r.tmax = min(tmax + 1.0e-3f, r.tmax);
  return true;
}

const MAX_LEVEL = 20u;
const BSP_LEAF = 3u;
var  <private> branch_node: array<vec2u, MAX_LEVEL>; 
var  <private> branch_ray: array<vec2f, MAX_LEVEL>;

fn int_trimesh(ray: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool{
  var branch_level = 0u;
  var near_node = 0u;
  var far_node = 0u;
  var t = 0.0f;
  var node = 0u;
  for(var i = 0u; i <= MAX_LEVEL; i ++){
    let tree_node = bspTree[node];
    // if tree_node.x has the 2 least significant bits set
    let node_axis_leaf = tree_node.x & 3u;
    if(node_axis_leaf == BSP_LEAF){
      // leaf found
      let node_count = tree_node.x >> 2u;
      let node_id = tree_node.y;
      var found = false;
      for(var j = 0u; j < node_count; j++){
        let obj_idx = treeIds[node_id + j];
        if(int_triangle(*ray, hit, obj_idx)){
          (*ray).tmax = (*hit).distance;
          found = true;
        }
      }
      if(found){ return true;}
      else if(0u == branch_level){
        // traversed whole tree, no intersection
        return false;
      }
      else {
        branch_level --;
        i = branch_node[branch_level].x;
        node = branch_node[branch_level].y;
        (*ray).tmin = branch_ray[branch_level].x;
        (*ray).tmax = branch_ray[branch_level].y;
        continue;
      }
    }
    let axis_direction = (*ray).direction[node_axis_leaf];
    let axis_origin = (*ray).origin[node_axis_leaf];
    if(axis_direction >= 0.0f){
      near_node = tree_node.z; // left node
      far_node = tree_node.w; // right
    } else {
      near_node = tree_node.w; // right
      far_node = tree_node.z; // left
    }
    let node_plane = bspPlanes[node];
    let denom = select(axis_direction, 1.0e-8f, abs(axis_direction) < 1.0e-8f);
    t = (node_plane - axis_origin) / denom;
    if(t > (*ray).tmax){
      node = near_node;
    } else if(t < (*ray).tmin) {
      node = far_node;
    } else {
      branch_node[branch_level].x = i;
      branch_node[branch_level].y = far_node;
      branch_ray[branch_level].x = t;
      branch_ray[branch_level].y = (*ray).tmax;
      branch_level ++;
      (*ray).tmax = t;
      node = near_node;
    }

  }
  return false;
}

fn int_triangle(ray: Ray, hit: ptr<function, HitInfo>, i: u32) -> bool {
  // verts is a u32 representing the vertices of the triangle 
  let verts = vert_indices[i].xyz;
  let v = array<vec3f, 3>(
    vert_attribs[verts[0]].pos.xyz,
    vert_attribs[verts[1]].pos.xyz,
    vert_attribs[verts[2]].pos.xyz
  );

  let norms = array<vec3f, 3>(
    vert_attribs[verts[0]].normal.xyz,
    vert_attribs[verts[1]].normal.xyz,
    vert_attribs[verts[2]].normal.xyz,
  );

  let e0 = v[1] - v[0];
  let e1 = v[2] - v[0];
  // crude normal for intersection calculation
  let n = cross(e0, e1);
  let omega_dot_n = dot(ray.direction, n);

  if(abs(omega_dot_n) > 1e-8){
    // make sure ray isnt parallel to triangle plane
    let origin_to_v0 = v[0] - ray.origin;
    let t = dot(origin_to_v0, n)/omega_dot_n;
    if(t <= ray.tmax && ray.tmin <= t){
      let partial = cross(origin_to_v0, ray.direction);

      let beta = dot(partial, e1) / omega_dot_n;
      if(beta >= 0){
        let gamma = -dot(partial, e0) / omega_dot_n;
        if(gamma >= 0 && (beta + gamma) <= 1){
          let matIndex = vert_indices[i].w;
          if (matIndex >= arrayLength(&materials)) {
            // Handle invalid material index
            (*hit).color_amb = vec3f(1.0);
          } else {
            let material = materials[matIndex];
            (*hit).color_diff = material.diffuse.rgb;
            (*hit).color_amb = material.emission.rgb;
          }
          (*hit).hit = true;
          (*hit).distance = t;
          // r(t) = o + t w
          (*hit).position = ray.origin + t * ray.direction;
          // find more precise normal for shading based on barycentric coordinates
          // weighted of the vertex normals
          let alpha = 1.0 - (beta + gamma);
          (*hit).normal = normalize(alpha * norms[0] + beta * norms[1] + gamma*norms[2]);
          (*hit).shader = shader_matte;
        }
      }
    }
  }

  return (*hit).hit;
}

struct Light {
  Li: vec3f,
  wi: vec3f,
  dist: f32,
}

fn sample_point_light(p: vec3f) -> Light {
  // return light info at the point p
  // light source is at (0, 1, 0) w/ intensity (pi, pi, pi)
  let x = vec3f(0, 2, 0);
  let I = vec3f(3.14159) / 10;
  let dist = distance(p, x); 
  let wi = normalize(x - p);
  let Li = I / pow(dist/1000, 2);// convert from mm to meters

  return Light(Li, wi, dist);
}

fn sample_directional_light(p: vec3f) -> Light {
  // sample from a directional light source
  // with direction: normalize(vec3f(-1.0))
  // and emitted radiance Le = (pi, pi, pi)
  let Le = vec3f(3.14159);
  let dist = 0.1; // very far away
  let light_direction = normalize(-dir_light_dir);
  return Light(Le, light_direction, dist);
}

// sample all triangles of area light
fn sample_trimesh_light(p: vec3f) -> Light {
  let numTriangles = arrayLength(&light_indices);
  // Light(Li, wi, dist);
  // these are averages weighted by area of triangle
  var lightAreaIntensity = vec3f(0);
  var totalArea = 0f;
  var lightCenter = vec3f(0);
  var lightNorm = vec3f(0);
  var lightDist = 0f;
  for(var idx = u32(0); idx < numTriangles; idx ++){
    // sample each triangle, find the
    // total emission and center point
    let index = light_indices[idx];
    let verts = vert_indices[index].xyz;
    // coords of the 3 corner vertices of light
    let v = array<vec3f, 3>(
      vert_attribs[verts[0]].pos.xyz,
      vert_attribs[verts[1]].pos.xyz,
      vert_attribs[verts[2]].pos.xyz
    );
    // vertex normals
    let norms = array<vec3f, 3>(
      vert_attribs[verts[0]].normal.xyz,
      vert_attribs[verts[1]].normal.xyz,
      vert_attribs[verts[2]].normal.xyz,
    );
    // in barycentric coordinates, center of triangle
    // is where alpha = beta = gamma = 0.333

    let e0 = v[1] - v[0];
    let e1 = v[2] - v[0];

    let areaCross = cross(e0, e1);
    // magnitude of vector = sqrt(dot(vector, vector))
    let area = length(areaCross) * 0.5; // area = 1/2 | e0 X e1 |
    totalArea += area;

    let center = (e0 + e1) * 0.333 + v[0];
    lightCenter += center * area;
    let norm = (norms[0] + norms[1] + norms[2]) * 0.3333;
    lightNorm += norm * area;

    // direction from point to light
    var Le = materials[vert_indices[index].w].emission.rgb;


    let dist = distance(p, center); // convert from mm to meters
    lightAreaIntensity += Le * area * pow(1/dist, 2);
  }
  // take area-weighted averages
  lightNorm = normalize(lightNorm / totalArea);
  lightCenter /= totalArea;

  let wi = normalize(lightCenter - p);
  let Li = dot(-wi, lightNorm) * lightAreaIntensity;

  return Light(Li, wi, distance(p, lightCenter));

}

fn check_shadow(pos: vec3f, lightdir: vec3f, lightdist: f32) -> bool{
  var lightray =  Ray(pos, lightdir, 10e-4, lightdist-10e-4);
  var lighthit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 1.0, 1.0, 0, vec2f(0.0));
  return int_scene(&lightray, &lighthit);
}


fn lambert(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
  var Lr = ((*hit).color_amb / 3.14159);
  let light = sample_trimesh_light((*hit).position);
  // distant area light, so just use one sample point for visibility chekc
  if(!check_shadow((*hit).position, light.wi, light.dist)){
    Lr += ((*hit).color_diff / (3.14159)) * light.Li * max(dot((*hit).normal, light.wi), 0.0);;
  }
  // use ambient light and reflected light
  return Lr;
}

fn phong(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
  let light = sample_directional_light((*hit).position);
  

  let wo = normalize((*r).origin - (*hit).position);
  let wr = reflect(-light.wi, (*hit).normal);

  var Lr = ((*hit).color_amb / 3.14159);
  if(!check_shadow((*hit).position, light.wi, light.dist)){
    Lr += 
    light.Li * 
    dot(light.wi, (*hit).normal) *
    (
      ((*hit).color_diff / 3.14159) + 
      (
        (*hit).color_specular * 
        ((*hit).shine + 2) * 0.15915494309 * // 1/2pi = 0.15915494309
        pow(max(dot(wo, wr), 0.0), (*hit).shine)
      )
    );
  }

  if(dot(Lr, Lr) < 0.5){
    return vec3f(0);
  }
  return Lr;
}

fn shade_refract(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
  // case 3 indicates a refractive material. The ray is re-cast but
      // deflected according to the relative indices or refraction
      (*r).origin = (*hit).position; // cast ray from intersection position
      (*r).tmin = 1e-4; // make sure we dont collide with the surface the ray is reflected off
      (*r).tmax = 10000; // reset tmax b/c casting a new ray
      (*hit).hit = false; // tell iterator to re-trace ray

      // bend the direction by the ratio of refractive indices
      // cos(in) = ( r \cdot n ) / (|r|*|n|)
      let cos_in = dot((*r).direction, (*hit).normal);
      let sin_sq_in = 1 - (cos_in*cos_in);
      let sin_sq_out = 
        ((*hit).refractive_ratio * (*hit).refractive_ratio)
        * sin_sq_in;
      let cos_sq_out = 1 - sin_sq_out;
      if(cos_sq_out < 0){
        // total internal reflection
        (*r).direction = reflect((*r).direction, (*hit).normal); // reflect the incoming ray about the surface normal
        return vec3f(0);
      }
      let n = (*hit).refractive_ratio;
      (*r).direction = 
        (*hit).normal * (n * cos_in - sqrt(cos_sq_out))
        + (*r).direction * n;
      return vec3f(0);
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f{
  switch (*hit).shader {
    case shader_matte { return lambert(r, hit);}
    case shader_reflect {
      // case 2 indicates a reflective material. we need to re-cast a ray from the reflected position
      // on the intersected surface. this means modifying the ray
      (*hit).hit = false; // tell iterator to re-trace ray
      (*r).origin = (*hit).position; // cast ray from intersection position
      (*r).direction = reflect((*r).direction, (*hit).normal); // reflect the incoming ray about the surface normal
      (*r).tmin = 1e-4; // make sure we dont collide with the surface the ray is reflected off
      (*r).tmax = 10000;
      return vec3f(0.0);
    } 
    case shader_refract {
      return shade_refract(r, hit);
    }
    case shader_phong {
      return phong(r, hit);
    }
    case shader_glossy {
      return phong(r, hit) + shade_refract(r, hit);
    }
    //case default { return -(*r).direction;}
    case default {return (*hit).color_diff + (*hit).color_amb;}
  }  
}

// the fragment defines the shader function run at each pixel
@fragment
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f{
  const bgcolor = vec4f(0.1, 0.3, 0.6, 0.9);
  const max_depth = 10;
  var result = vec3f(0.0);
  // iterate over each sub-pixel position
  let num_subpixels = uniforms_int.subdivisions * uniforms_int.subdivisions;
  let inv_subpixels = 1.0 / f32(num_subpixels);
  for(var sub_idx = u32(0); sub_idx < num_subpixels; sub_idx ++){
    var hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 1.0, 1.0, 0, vec2f(0.0));
    var thisResult = vec3f(0.0);
    let thisSub = subpixels[sub_idx]; // add jittered subpixel coordinates
    var ipcoords = vec2f((coords.x + thisSub.x)*uniforms_f.aspect*0.5, (coords.y + thisSub.y)*0.5);
    var r = get_camera_ray(ipcoords); 
    for(var i = 0; i < max_depth; i ++){
      if(int_scene(&r, &hit)){
        thisResult += shade(&r, &hit);
        if(hit.hit){
          break;
        }
        if(dot(thisResult, thisResult) >= 0.99){
          // save some computation if saturated already
          break;
        }
      } else {
        thisResult += bgcolor.rgb;
        break;
      }
    }
    result += inv_subpixels * thisResult;
  }
  return vec4f(result, bgcolor.a); 
  //return vec4f(f32(uniforms_int.subdivisions) / 10.0, 0, 0, 1.0);
}