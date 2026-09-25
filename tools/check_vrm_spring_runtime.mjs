// Usage: node tools/check_vrm_spring_runtime.mjs candidate.vrm npm-prefix
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
// Use the public ESM entries from an isolated npm prefix.
async function loadPackage(name){
 const dir=path.resolve(process.argv[3], 'node_modules', name);
 const pkg=JSON.parse(fs.readFileSync(path.join(dir,'package.json')));
 const entry=pkg.exports?.['.']?.import ?? pkg.module ?? pkg.main;
 if(typeof entry!=='string')throw Error(`No public ESM entry for ${name}`);
 return import(pathToFileURL(path.join(dir,entry)).href);
}
const THREE=await loadPackage('three');
const {VRMSpringBoneLoaderPlugin}=await loadPackage('@pixiv/three-vrm-springbone');
const input=process.argv[2];
const data=fs.readFileSync(input);
const json=JSON.parse(data.subarray(20,20+data.readUInt32LE(12)).toString());
const nodes=json.nodes.map(n=>{
 const o=new THREE.Object3D();o.name=n.name??'';
 if(n.matrix){o.matrix.fromArray(n.matrix);o.matrix.decompose(o.position,o.quaternion,o.scale);}
 else{if(n.translation)o.position.fromArray(n.translation);if(n.rotation)o.quaternion.fromArray(n.rotation);if(n.scale)o.scale.fromArray(n.scale);}
 o.updateMatrix();return o;
});
json.nodes.forEach((n,i)=>(n.children??[]).forEach(c=>nodes[i].add(nodes[c])));
const scene=new THREE.Scene();nodes.filter(n=>!n.parent).forEach(n=>scene.add(n));scene.updateMatrixWorld(true);
const parser={json,getDependencies:async type=>{if(type!=='node')throw Error(type);return nodes;}};
const gltf={parser,scene,userData:{}};
const warnings=[];const warn=console.warn;console.warn=(...a)=>warnings.push(a.join(' '));
await new VRMSpringBoneLoaderPlugin(parser).afterRoot(gltf);
const manager=gltf.userData.vrmSpringBoneManager;
if(!manager)throw Error('Spring import failed');
const human=json.extensions.VRMC_vrm.humanoid.humanBones;
const leg=nodes[human.leftUpperLeg.node],rest=leg.quaternion.clone();
for(let f=0;f<180;f++){
 const t=Math.sin(Math.PI*f/179)*Math.PI/6;
 leg.quaternion.copy(rest).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(t,t/2,t)));
 leg.updateMatrix();scene.updateMatrixWorld(true);manager.update(1/60);
 for(const n of nodes)if(!n.matrixWorld.elements.every(Number.isFinite))throw Error('Nonfinite pose');
}
console.warn=warn;
const report={input,runtime:'Official three-vrm-springbone loader and solver from supplied npm prefix',joints:manager.joints.size,frames:180,warnings,scope:'Exported node hierarchy and official spring loader/solver; no mesh rendering or mesh contact validation'};
fs.writeFileSync(input+'.three-vrm.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
