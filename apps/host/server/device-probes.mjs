/** Physical topology and GPU sensors; unavailable values stay null. */
import { readFile, readdir, realpath } from 'node:fs/promises';
import { probe, round } from './util.mjs';
import { windowsGpuReading } from './windows-gpu.mjs';

/** Unified GPUs share one system memory pool; dedicated GPUs each have their own. */
export function gpuMemoryTotals(devices, readings) {
  const id = gpu => gpu.id ?? `${gpu.vendor}:${gpu.index}`;
  const values = devices.map(gpu => readings.find(r => r.id === id(gpu))?.memoryUsedGb);
  const total = devices.filter(g => !g.unified).reduce((n,g) => n + g.vramGb,0)
    + Math.max(0,...devices.filter(g => g.unified).map(g => g.vramGb));
  if (!values.length || values.some(v => typeof v !== 'number' || !Number.isFinite(v))) return {used:null,total};
  const used = values.filter((_,i) => !devices[i].unified).reduce((n,v) => n+v,0)
    + Math.max(0,...values.filter((_,i) => devices[i].unified));
  return {used,total};
}

export function parseCpuTopology(raw, model, arch, ghz = 0, names = new Map()) {
  const sockets = new Map();
  for (const line of raw.split('\n')) {
    if (!line.trim() || line.startsWith('#')) continue;
    const [cpu, core, socket] = line.split(',').map(v=>v.trim());
    if (!/^\d+$/.test(cpu) || !/^\d+$/.test(core) || !/^\d+$/.test(socket)) continue;
    const entry = sockets.get(socket) ?? { cores:new Set(), threads:new Set() };
    entry.cores.add(core); entry.threads.add(cpu); sockets.set(socket,entry);
  }
  return [...sockets].map(([socket,s])=>({id:`socket-${socket}`,socket,model:names.get(socket)??model,cores:s.cores.size,threads:s.threads.size,logicalIds:[...s.threads].map(Number),arch,ghz}));
}
export function parseWindowsProcessors(raw, arch) {
  try {
    const parsed = JSON.parse(raw), rows = Array.isArray(parsed)?parsed:[parsed];
    return rows.filter(r=>r?.DeviceID&&Number(r.NumberOfCores)>0&&Number(r.NumberOfLogicalProcessors)>0).map(r=>({id:String(r.DeviceID),socket:String(r.SocketDesignation??r.DeviceID),model:String(r.Name??'Unknown processor').trim(),cores:Number(r.NumberOfCores),threads:Number(r.NumberOfLogicalProcessors),arch,ghz:round(Number(r.MaxClockSpeed??0)/1000,2)}));
  } catch { return []; }
}
const number = value => value!=null&&String(value).trim()!==''&&Number.isFinite(Number(value))?Number(value):null;
export function parseNvidiaDevices(raw) {
  return raw.split('\n').flatMap(line=>{
    const [index,uuid,bus,name,memory,driver]=line.split(',').map(s=>s.trim());
    if (!uuid?.startsWith('GPU-') || !name) return [];
    return [{id:uuid,index:Number(index),sensorIndex:Number(index),busId:bus?.toLowerCase(),vendor:'nvidia',name,vramGb:round((number(memory)??0)/1024,1),unified:false,backend:'cuda',driver}];
  });
}
/** A percentage reading; "[N/A]" and out-of-range values are no reading. */
const percentOf = v => { const n=number(v); return n!==null&&n>=0&&n<=100?n:null; };
export function parseNvidiaReadings(raw) {
  return raw.split('\n').flatMap(line=>{
    const [id,temp,load,memory,power,limit,gen,maxGen,width,fan,clock]=line.split(',').map(s=>s.trim());
    if (!id?.startsWith('GPU-')) return [];
    const t=number(temp),u=number(load),m=number(memory);
    const positive = v => { const n=number(v); return n!==null&&n>=0?n:null; };
    return [{id,temperatureC:t!==null&&t>=0&&t<=150?t:null,utilizationPct:u!==null&&u>=0&&u<=100?u:null,memoryUsedGb:m!==null&&m>=0?round(m/1024,2):null,powerW:positive(power),powerLimitW:positive(limit),pcieGen:positive(gen),pcieGenMax:positive(maxGen),pcieWidth:positive(width),pcieRxMbs:null,fanPct:percentOf(fan),clockMhz:positive(clock)}];
  });
}
export function parsePcieThroughput(raw) {
  const values=new Map();
  for(const line of raw.split('\n')) {
    if(line.trim().startsWith('#'))continue;
    const fields=line.trim().split(/\s+/),index=number(fields[0]),rx=number(fields[1]);
    if(fields.length===3&&index!==null&&Number.isInteger(index)&&index>=0&&rx!==null&&rx>=0)values.set(index,rx);
  }
  return values;
}
async function read(path) { try {return (await readFile(path,'utf8')).trim();} catch {return '';} }
async function list(path) {try{return await readdir(path);}catch{return [];}}
/** A server's remote-console chip (ASPEED), a virtual display or a framebuffer is not a graphics card: no maker we know and no memory of its own. */
export function isComputeAdapter(vendor,vramBytes) { return vendor!=='unknown'||vramBytes>0; }

/** The PCI bus id in one spelling: nvidia-smi writes "00000000:03:00.0", the sysfs device folder is named "0000:03:00.0". */
export function normalizeBusId(bus) {
  const text = String(bus ?? '').trim().toLowerCase();
  return text ? text.replace(/^0{4,}:/, '0000:') : null;
}
/** A path segment that names a PCI device ("0000:03:00.0"), the same shape serve/telemetry.py checks. */
export function isPciBusName(part) {
  const s = String(part ?? '');
  return s.length === 12 && s[4] === ':' && s[7] === ':' && s[10] === '.' && '01234567'.includes(s[11]) &&
    /^[0-9a-f]{4}$/.test(s.slice(0,4)) && /^[0-9a-f]{2}$/.test(s.slice(5,7)) && /^[0-9a-f]{2}$/.test(s.slice(8,10));
}
/**
 * The PCI bus id inside a resolved sysfs path: the card's own device folder is the last PCI-named part. An earlier part can
 * be a bridge hop with the same shape ("0000:00:01.1"), so the first match is not the card.
 */
export function busIdFromPath(path) { return String(path ?? '').split('/').filter(isPciBusName).at(-1) ?? null; }

/** One KFD topology node's `properties` text -> what the engine's numbering needs: {bdf, gfx, simds, minor}. */
export function parseKfdNode(raw) {
  const props = {};
  for (const line of String(raw ?? '').split('\n')) {
    const [key,...rest] = line.trim().split(/\s+/);
    if (key) props[key] = rest.join(' ');
  }
  return {bdf:props.bdf ?? null,gfx:number(props.gfx_target_version) ?? 0,simds:number(props.simd_count) ?? 0,minor:props.drm_render_minor ?? null};
}

/**
 * The cards in the order the engine numbers them (HIP): the KFD topology's GPU nodes in node order, the CPU nodes skipped -
 * the walk setup.py's `amd_gpus()` and serve/telemetry.py's `amd_device_dir()` do. The host's own `index` is a position in
 * its own list, and an integrated Radeon or an adapter the probe drops shifts that list, so the engine's number has to be
 * resolved to a physical card rather than assumed. Null when the order cannot be built (no amdgpu, a container that hides
 * /sys, a GPU node whose card cannot be identified). Pure: `nodes` are parsed KFD node properties in node order.
 */
export function kfdGpuOrder(nodes) {
  const gpus = nodes.filter(n => n.gfx > 0 && n.simds > 0);
  if (!gpus.length) return null;
  const order = gpus.map(n => normalizeBusId(n.bdf));
  return order.some(b => b === null) ? null : order;
}
/** The same order read from sysfs, a node without a `bdf` identified through its render node. */
export async function readKfdGpuOrder(sysfs = '/sys') {
  const base = `${sysfs}/class/kfd/kfd/topology/nodes`;
  const entries = await list(base);
  if (!entries.length) return null;
  const nodes = [];
  for (const name of entries.filter(n => /^\d+$/.test(n)).sort((a,b) => Number(a)-Number(b))) {
    const node = parseKfdNode(await read(`${base}/${name}/properties`));
    if (!node.bdf && node.minor) node.bdf = busIdFromPath(await realpath(`${sysfs}/class/drm/renderD${node.minor}`));
    nodes.push(node);
  }
  return kfdGpuOrder(nodes);
}

/**
 * The number the engine gives each card, kept as `engineIndex` (null when it cannot be resolved). nvidia-smi's own index is
 * the engine's number (it numbers by PCI bus); an amdgpu card's is its position in `kfdOrder`; a card the engine's numbering
 * does not reach (an Intel or Apple GPU, a card outside the KFD tree) stays null, and its engine reading is not claimed for it.
 */
export function assignEngineNumbers(devices,kfdOrder = null) {
  return devices.map(g => {
    if (g.vendor==='nvidia') return {...g,engineIndex:g.sensorIndex ?? g.index};
    const at = kfdOrder ? kfdOrder.indexOf(normalizeBusId(g.busId)) : -1;
    return {...g,engineIndex:at>=0?at:null};
  });
}

/** The engine's number resolved to this card, or null when it is not known. */
export const engineIndex = gpu => Number.isInteger(gpu.engineIndex) ? gpu.engineIndex : null;

/** A card Strata cannot run a model on: an integrated chip with a carve-out under 2 GB beside a real card. Mirrored in src/lib/monitor-hardware.ts. */
export function smallGpu(gpu) { return gpu.backend === 'vulkan' && (gpu.vendor === 'amd' || gpu.vendor === 'intel') && (gpu.vramGb ?? 0) < 2 && !gpu.unified; }

/**
 * The card an engine reading for card `number` belongs to. The engine numbers the physical cards (HIP's KFD order, nvidia-smi's
 * order by PCI bus); `index` is a position in the host's own list, and an integrated chip or an adapter the probe drops shifts
 * that list, so the two agree only when the lists happen to come out in the same order. Identity first. With no identity for any
 * card, a reading can only belong to a card the engine can run a model on - one such card is the card; else the number has to name
 * exactly one card. Otherwise nothing is claimed, so a tile says unavailable instead of showing another card's numbers. Mirrored in
 * src/lib/monitor-hardware.ts; tests/monitor-dashboard.test.mjs runs the same table through both.
 */
export function engineDevice(devices,number) {
  const known = devices.filter(d => engineIndex(d) !== null);
  const hit = known.find(d => engineIndex(d) === number);
  if (hit) return hit;
  if (known.length) return null;
  const canRun = devices.filter(d => !smallGpu(d));
  if (devices.length > 1 && canRun.length === 1) return canRun[0];
  const same = devices.filter(d => d.index === number);
  return same.length === 1 ? same[0] : null;
}
/**
 * The temperature of an amdgpu card from its hwmon inputs [{label, milli}]: the edge sensor (what nvtop and rocm-smi show), else the
 * junction, else the first one that has a plausible value. Cards such as the Instinct line have no edge sensor. 0 is no reading.
 */
export function pickGpuTemperature(inputs) {
  const ok=inputs.filter(i=>i.milli!==null&&i.milli>0&&i.milli<=150000);
  const first=re=>ok.find(i=>re.test(i.label??''));
  const chosen=first(/edge|^gpu/i)??ok.find(i=>!i.label)??first(/junction|hotspot/i)??ok.find(i=>!/mem/i.test(i.label??''));
  return chosen?chosen.milli/1000:null;
}
export async function detectDrmGpus() {
  const result=[];
  /* The order is the physical order, not readdir's: the host's `index` is a position in this list, and an unsorted read
     would make it change between probes. */
  for(const card of (await list('/sys/class/drm')).filter(n=>/^card\d+$/.test(n)).sort((a,b) => Number(a.slice(4))-Number(b.slice(4)))) {
    const device=`/sys/class/drm/${card}/device`;
    let resolved;try{resolved=await realpath(device);}catch{continue;}
    const busId=normalizeBusId(busIdFromPath(resolved)),vendorId=await read(`${device}/vendor`);
    const vendor={'0x1002':'amd','0x8086':'intel','0x10de':'nvidia'}[vendorId]??'unknown';
    const name=await probe('lspci',['-s',busId]);
    const bytes=number(await read(`${device}/mem_info_vram_total`));
    if(!isComputeAdapter(vendor,bytes))continue;
    result.push({id:`drm:${busId}`,index:result.length,busId,vendor,name:name.replace(/^[^ ]+\s+[^:]+:\s*/,'')||`${vendor.toUpperCase()} GPU`,vramGb:bytes===null?0:round(bytes/1024**3,1),unified:vendor==='intel'&&!(bytes>0),backend:'vulkan',sensorPath:device});
  }
  return result;
}
/** `transfer: false` skips the PCIe throughput sample (a second nvidia-smi); it only means something while a model runs. */
export async function readGpuTelemetry(gpus, { transfer: wantTransfer = true, run = probe } = {}) {
  const now=new Date().toISOString();
  const hasNvidia=gpus.some(g=>g.vendor==='nvidia');
  let raw=hasNvidia?await run('nvidia-smi',['--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used,power.draw,power.limit,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,fan.speed,clocks.current.graphics','--format=csv,noheader,nounits'],2000):'';
  // Older drivers can reject a field; retain their basic sensors in that case.
  if(hasNvidia&&!raw)raw=await run('nvidia-smi',['--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits'],2000);
  const nvidia=new Map(parseNvidiaReadings(raw).map(r=>[r.id,r]));
  const transfer=hasNvidia&&wantTransfer?parsePcieThroughput(await run('nvidia-smi',['dmon','-s','t','-c','1'],1500)):new Map();
  return Promise.all(gpus.map(async gpu=>{
    let reading=nvidia.get(gpu.id)??{id:gpu.id??`${gpu.vendor}:${gpu.index}`,temperatureC:null,utilizationPct:null,memoryUsedGb:null,powerW:null,powerLimitW:null,pcieGen:null,pcieGenMax:null,pcieWidth:null,pcieRxMbs:null};
    if(gpu.vendor==='nvidia')reading={...reading,pcieRxMbs:transfer.get(gpu.sensorIndex??gpu.index)??null};
    if(gpu.luid&&!nvidia.has(gpu.id)) {
      const counters=windowsGpuReading(gpu.luid);
      if(counters)reading={...reading,utilizationPct:counters.utilizationPct,memoryUsedGb:round(counters.memoryUsedGb,2),temperatureC:counters.temperatureC??null};
    }
    if(gpu.sensorPath&&!nvidia.has(gpu.id)) {
      let temperature=null,power=null,powerLimit=null,fanPwm=null,clockHz=null;
      for(const hwmon of await list(`${gpu.sensorPath}/hwmon`)) {
        const folder=`${gpu.sensorPath}/hwmon/${hwmon}`,inputs=(await list(folder)).filter(n=>/^temp\d+_input$/.test(n));
        power ??= number(await read(`${folder}/power1_average`)) ?? number(await read(`${folder}/power1_input`));
        powerLimit ??= number(await read(`${folder}/power1_cap`));
        fanPwm ??= number(await read(`${folder}/pwm1`));          // 0-255, the fan's duty as the amdgpu driver reports it
        clockHz ??= number(await read(`${folder}/freq1_input`));  // the graphics clock in Hz
        const sensors=[];
        for(const input of inputs)sensors.push({label:await read(`${folder}/${input.replace('_input','_label')}`),milli:number(await read(`${folder}/${input}`))});
        temperature=pickGpuTemperature(sensors);
        if(temperature!==null)break;
      }
      const utilization=number(await read(`${gpu.sensorPath}/gpu_busy_percent`)),bytes=number(await read(`${gpu.sensorPath}/mem_info_vram_used`));
      const gen=path=>read(path).then(raw=>({2.5:1,5:2,8:3,16:4,32:5,64:6}[Number.parseFloat(raw)]??null));
      const [pcieGen,pcieGenMax,width]=await Promise.all([gen(`${gpu.sensorPath}/current_link_speed`),gen(`${gpu.sensorPath}/max_link_speed`),read(`${gpu.sensorPath}/current_link_width`)]);
      reading={...reading,fanPct:fanPwm!==null&&fanPwm>=0&&fanPwm<=255?Math.round(fanPwm/255*100):null,clockMhz:clockHz!==null&&clockHz>0?Math.round(clockHz/1e6):null,temperatureC:temperature,utilizationPct:utilization!==null&&utilization>=0&&utilization<=100?utilization:null,memoryUsedGb:bytes!==null&&bytes>=0?round(bytes/1024**3,2):null,powerW:power!==null&&power>=0?power/1e6:null,powerLimitW:powerLimit!==null&&powerLimit>=0?powerLimit/1e6:null,pcieGen,pcieGenMax,pcieWidth:number(width)};
    }
    return {...reading,sampledAt:now,source:Object.entries(reading).some(([key,value])=>key!=='id'&&typeof value==='number')?'sensor':'none'};
  }));
}
