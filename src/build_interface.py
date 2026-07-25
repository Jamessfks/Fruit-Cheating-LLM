"""Generate the Fruit-Drama input-system interface (interface/index.html) from
data/fruit_bible.json + prompts/system_fruit.txt.

Self-contained (no network, no libs) so it opens by double-click. The user picks
fruits, sets each one's gender/stage/role/disposition/character + a context
prompt, and it assembles the full system+user prompt for the fine-tuned LLM.
Also writes interface/_widget.html (body-only fragment for inline preview).
"""
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = json.load(open(os.path.join(ROOT, "data", "fruit_bible.json"), encoding="utf-8"))
SYS = open(os.path.join(ROOT, "prompts", "system_fruit.txt"), encoding="utf-8").read().strip()
OUT_DIR = os.path.join(ROOT, "interface")
os.makedirs(OUT_DIR, exist_ok=True)

ROLES = ["Yoga instructor", "Gym bro / trainer", "CEO / mogul", "Housewife / househusband",
         "Detective", "Doctor / surgeon", "Lawyer", "Chef", "Influencer", "Bartender",
         "Politician", "Priest / pastor", "Teacher", "Nurse", "Model", "Retired pro athlete",
         "Mob boss", "Heir / heiress", "Maid / butler", "Bodyguard", "Artist", "Pop star",
         "Nanny", "Real-estate mogul", "Wedding planner", "Pilot", "Farmer", "Fortune teller",
         "Bride", "Groom", "Mistress", "Mother-in-law", "The ex", "Long-lost twin",
         "Con artist", "Socialite", "Personal trainer", "Bouncer", "Barista", "Fashion designer"]
DISPOSITIONS = ["Hot and arrogant", "Shy and sweet", "Cold and calculating", "Warm but naive",
                "Secretly rich", "Secretly broke", "Vengeful", "Heartbroken", "Ambitious",
                "Jealous", "Loyal to a fault", "Two-faced", "Seductive", "Insecure", "Ruthless",
                "Protective", "Manipulative", "Righteous", "Reckless", "Hiding a secret",
                "Newly pregnant", "Recently betrayed", "Power-hungry", "Desperate for approval",
                "Cool and detached", "Disciplined", "Haunted by the past", "Overconfident"]
CHARACTERS = ["Protagonist", "Antagonist / Villain", "Temptress / Seducer", "Innocent victim",
              "Matriarch / Patriarch", "The Heir", "The Betrayer", "The Peacemaker",
              "The Schemer", "Comic relief", "The Wildcard", "The Mentor", "The Outsider",
              "The Golden child", "The Black sheep", "The Gossip", "The Enforcer",
              "The Martyr", "The Trophy", "The Underdog"]

BODY = r"""
<style>
  :root{--plum:#5B2A86;--plum2:#7e46b0;--bg:#faf7fd;--card:#ffffff;--ink:#241033;--muted:#7a6b8a;--line:#e7ddf2;--gold:#F9A825;}
  *{box-sizing:border-box}
  .fd-wrap{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:var(--ink);max-width:1100px;margin:0 auto;padding:8px 14px 40px}
  .fd-wrap h1{font-size:26px;margin:6px 0 2px}
  .fd-wrap .sub{color:var(--muted);margin:0 0 16px;font-size:14px}
  .fd-bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
  .fd-bar input[type=text]{flex:1;min-width:180px;padding:9px 12px;border:1px solid var(--line);border-radius:10px;font-size:14px}
  .fd-btn{background:var(--plum);color:#fff;border:0;padding:9px 15px;border-radius:10px;font-size:14px;cursor:pointer;font-weight:600}
  .fd-btn:hover{background:var(--plum2)}
  .fd-btn.ghost{background:#efe7f7;color:var(--plum)}
  .fd-count{font-size:13px;color:var(--muted)}
  .fd-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:8px;max-height:300px;overflow:auto;padding:6px;border:1px solid var(--line);border-radius:12px;background:#fff}
  .fd-chip{border:1px solid var(--line);border-radius:12px;background:#fff;padding:8px 4px;text-align:center;cursor:pointer;transition:.12s;user-select:none}
  .fd-chip:hover{border-color:var(--plum2);transform:translateY(-1px)}
  .fd-chip.sel{background:var(--plum);color:#fff;border-color:var(--plum)}
  .fd-chip .em{font-size:26px;line-height:1.1}
  .fd-chip .nm{font-size:11px;margin-top:2px;font-weight:600}
  .fd-chip .g{font-size:10px;opacity:.7}
  .fd-sec-title{font-weight:700;margin:20px 0 8px;font-size:15px;display:flex;align-items:center;gap:8px}
  .fd-cast{display:grid;grid-template-columns:1fr;gap:10px}
  .fd-cc{border:1px solid var(--line);border-left:4px solid var(--plum);border-radius:12px;background:var(--card);padding:12px}
  .fd-cc .top{display:flex;align-items:center;gap:8px;margin-bottom:8px}
  .fd-cc .top .em{font-size:24px}
  .fd-cc .top .nm{font-weight:700;font-size:15px}
  .fd-cc .top .rm{margin-left:auto;background:#f7eef0;color:#a12;border:0;border-radius:8px;padding:5px 9px;cursor:pointer;font-size:12px}
  .fd-fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
  .fd-fields label{font-size:11px;color:var(--muted);display:block;margin-bottom:2px}
  .fd-fields select,.fd-fields input{width:100%;padding:7px 8px;border:1px solid var(--line);border-radius:9px;font-size:13px;background:#fff}
  .fd-trait{font-size:12px;color:var(--plum);background:#f6f0fc;border-radius:8px;padding:7px 9px;margin-top:8px}
  .fd-trait b{color:var(--ink)}
  .fd-context{width:100%;min-height:80px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;font-size:14px;font-family:inherit}
  .fd-opts{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:10px 0}
  .fd-opts input,.fd-opts select{padding:7px 9px;border:1px solid var(--line);border-radius:9px;font-size:13px}
  .fd-out{white-space:pre-wrap;background:#1f1030;color:#f3e9ff;border-radius:12px;padding:14px;font-size:12.5px;line-height:1.5;max-height:360px;overflow:auto;font-family:ui-monospace,Menlo,Consolas,monospace}
  .fd-empty{color:var(--muted);font-size:13px;padding:12px;border:1px dashed var(--line);border-radius:12px;text-align:center}
  .fd-foot{color:var(--muted);font-size:12px;margin-top:8px}
</style>

<div class="fd-wrap">
  <h1>🍑 Fruit Drama — Story Builder</h1>
  <p class="sub">Pick your cast, set each fruit's gender · life-stage · role · disposition · character, add a context, and generate the prompt for the fruit-drama LLM.</p>

  <div class="fd-bar">
    <input type="text" id="fdSearch" placeholder="Search 64 fruits… (banana, peach, eggplant)">
    <button class="fd-btn ghost" id="fdExample">Load example</button>
    <button class="fd-btn ghost" id="fdClear">Clear</button>
    <span class="fd-count" id="fdCount">0 selected</span>
  </div>
  <div class="fd-grid" id="fdGrid"></div>

  <div class="fd-sec-title">🎭 Your cast <span class="fd-count" id="fdCastHint"></span></div>
  <div id="fdCast" class="fd-cast"><div class="fd-empty">No fruits selected yet — click fruits above to add them.</div></div>

  <div class="fd-sec-title">🎬 Context, location &amp; setting</div>
  <textarea class="fd-context" id="fdContext" placeholder="e.g. A luxury wellness retreat in the hills during the annual charity gala; old rivalries resurface and a paternity secret is about to detonate."></textarea>
  <div class="fd-opts">
    <label>Episode # <input type="number" id="fdEp" value="1" min="1" style="width:70px"></label>
    <label>Tone
      <select id="fdTone">
        <option>serious &amp; over-the-top telenovela</option>
        <option>dark &amp; twisty true-crime</option>
        <option>campy &amp; absurd comedy</option>
        <option>tragic &amp; emotional</option>
      </select>
    </label>
    <button class="fd-btn" id="fdGen">✨ Generate prompt</button>
  </div>

  <div class="fd-sec-title">🎬 Generate live from your fine-tuned model</div>
  <div class="fd-opts">
    <label>Model API <input type="text" id="fdEndpoint" value="http://spark-2e6c:8000/v1/chat/completions" style="min-width:300px"></label>
    <label>Model <input type="text" id="fdModel" value="fruit" style="width:80px"></label>
    <button class="fd-btn" id="fdRun">▶ Generate episode</button>
    <span class="fd-count" id="fdStatus"></span>
  </div>
  <div class="fd-out" id="fdEpisode">Your generated episode will appear here (needs the vLLM server running on the Spark).</div>

  <div class="fd-sec-title">📝 Prompt for the LLM
    <button class="fd-btn ghost" id="fdCopy" style="margin-left:auto">Copy prompt</button>
    <button class="fd-btn ghost" id="fdCopyJson">Copy JSON</button>
  </div>
  <div class="fd-out" id="fdOut">Select fruits and press “Generate prompt”.</div>
  <p class="fd-foot">This is the <b>input system</b>: it composes the system + user message. Paste it into the fine-tuned fruit-drama model (or wire this page to its API). System prompt and all 64 fruit profiles are embedded — works offline.</p>
</div>

<script>
const FRUITS = __FRUITS__;
const ROLES = __ROLES__, DISPOSITIONS = __DISPOSITIONS__, CHARACTERS = __CHARACTERS__;
const SYSTEM = __SYSPROMPT__;
const STAGES = [
  {k:'growth', label:'Growth (green / young)'},
  {k:'prime', label:'Prime (ripe / peak)'},
  {k:'ripening', label:'Ripening (the fall begins)'},
  {k:'senescence', label:'Senescence (rotten / failure)'},
];
const byName = Object.fromEntries(FRUITS.map(f=>[f.name,f]));
const sel = new Map(); // name -> {gender,stage,role,disp,char,note}

function defGender(f){ return f.gender==='Male-dominant' ? 'Male' : 'Female'; }

function renderGrid(){
  const q = (document.getElementById('fdSearch').value||'').toLowerCase();
  const g = document.getElementById('fdGrid'); g.innerHTML='';
  FRUITS.filter(f=>f.name.toLowerCase().includes(q)||f.archetype.toLowerCase().includes(q)).forEach(f=>{
    const d=document.createElement('div');
    d.className='fd-chip'+(sel.has(f.name)?' sel':'');
    const badge = f.gender==='Male-dominant'?'♂':f.gender==='Female-dominant'?'♀':'⚧';
    d.innerHTML=`<div class="em">${f.emoji}</div><div class="nm">${f.name}</div><div class="g">${badge}</div>`;
    d.onclick=()=>toggle(f.name);
    g.appendChild(d);
  });
}
function toggle(name){
  if(sel.has(name)) sel.delete(name);
  else { const f=byName[name]; sel.set(name,{gender:defGender(f),stage:'prime',role:'',disp:'',char:'',note:''}); }
  renderGrid(); renderCast();
}
function opts(list, cur){ return list.map(x=>`<option${x===cur?' selected':''}>${x}</option>`).join(''); }
function dataOpts(list){ return list.map(x=>`<option value="${x.replace(/"/g,'&quot;')}">`).join(''); }

function renderCast(){
  const c=document.getElementById('fdCast');
  document.getElementById('fdCount').textContent = sel.size+' selected';
  document.getElementById('fdCastHint').textContent = sel.size?('· '+sel.size+' character'+(sel.size>1?'s':'')):'';
  if(!sel.size){ c.innerHTML='<div class="fd-empty">No fruits selected yet — click fruits above to add them.</div>'; return; }
  c.innerHTML='';
  for(const [name,st] of sel){
    const f=byName[name];
    const card=document.createElement('div'); card.className='fd-cc';
    card.innerHTML=`
      <div class="top"><span class="em">${f.emoji}</span><span class="nm">${f.name} <span style="color:var(--muted);font-weight:400">· ${f.example}</span></span>
        <button class="rm" data-n="${name}">✕ remove</button></div>
      <div class="fd-fields">
        <div><label>Gender</label><select data-k="gender"><option${st.gender==='Female'?' selected':''}>Female</option><option${st.gender==='Male'?' selected':''}>Male</option></select></div>
        <div><label>Life stage</label><select data-k="stage">${STAGES.map(s=>`<option value="${s.k}"${s.k===st.stage?' selected':''}>${s.label}</option>`).join('')}</select></div>
        <div><label>Role</label><input list="fdRoles" data-k="role" value="${st.role.replace(/"/g,'&quot;')}" placeholder="e.g. Yoga instructor"></div>
        <div><label>Disposition</label><input list="fdDisp" data-k="disp" value="${st.disp.replace(/"/g,'&quot;')}" placeholder="e.g. Hot and arrogant"></div>
        <div><label>Dramatic character</label><input list="fdChars" data-k="char" value="${st.char.replace(/"/g,'&quot;')}" placeholder="e.g. Temptress"></div>
        <div><label>Custom note</label><input data-k="note" value="${st.note.replace(/"/g,'&quot;')}" placeholder="secret, twist hook…"></div>
      </div>
      <div class="fd-trait" data-trait></div>`;
    card.querySelector('.rm').onclick=()=>toggle(name);
    card.querySelectorAll('[data-k]').forEach(el=>{
      el.oninput=()=>{ st[el.dataset.k]=el.value; if(el.dataset.k==='stage') updTrait(card,f,st); };
    });
    c.appendChild(card); updTrait(card,f,st);
  }
}
function updTrait(card,f,st){
  card.querySelector('[data-trait]').innerHTML =
    `<b>${STAGES.find(s=>s.k===st.stage).label}:</b> ${f[st.stage]} &nbsp;·&nbsp; <b>Cheats:</b> ${f.cheating}`;
}

function assemble(){
  if(!sel.size) return {text:'Select at least one fruit first.', json:null};
  const ep=document.getElementById('fdEp').value||1;
  const tone=document.getElementById('fdTone').value;
  const ctx=(document.getElementById('fdContext').value||'').trim() || '(no context given — invent a juicy one)';
  let cast=[]; let i=1;
  for(const [name,st] of sel){
    const f=byName[name];
    const stageLabel=STAGES.find(s=>s.k===st.stage).label;
    cast.push(
`${i}. ${f.emoji} ${f.name} ("${f.example}") — ${st.gender}, ${stageLabel}.
   Stage persona: ${f[st.stage]}
   Personality: ${f.personality} Character: ${f.character}
   Cheating/betrayal pattern: ${f.cheating}
   Role: ${st.role||'(you choose)'} · Disposition: ${st.disp||'(you choose)'} · Dramatic function: ${st.char||f.archetype}${st.note?('\n   Note: '+st.note):''}`);
    i++;
  }
  const user =
`Write ONE 6-scene AI fruit-drama episode (Episode ${ep}) that intertwines the cast below into a single, mind-blowing storyline with a shocking final twist. Tone: ${tone}.

CAST:
${cast.join('\n')}

CONTEXT / LOCATION / SETTING:
${ctx}

Connect these characters through secrets, betrayals, affairs and rivalries drawn from their cheating patterns and current life-stages. Escalate every scene and end on a jaw-dropping cliffhanger. Follow the exact 6-scene format with a NARRATION line and a VISUAL line for each scene.`;
  const text = `=== SYSTEM ===\n${SYSTEM}\n\n=== USER ===\n${user}`;
  const json = JSON.stringify({messages:[{role:'system',content:SYSTEM},{role:'user',content:user}]}, null, 2);
  return {text, user, json};
}

function copy(t){ navigator.clipboard && navigator.clipboard.writeText(t); }

document.getElementById('fdSearch').oninput=renderGrid;
document.getElementById('fdClear').onclick=()=>{sel.clear();renderGrid();renderCast();document.getElementById('fdOut').textContent='Select fruits and press “Generate prompt”.';};
document.getElementById('fdGen').onclick=()=>{ const a=assemble(); document.getElementById('fdOut').textContent=a.text; window._fdA=a; };
document.getElementById('fdCopy').onclick=()=>{ const a=window._fdA||assemble(); copy(a.text); };
document.getElementById('fdCopyJson').onclick=()=>{ const a=window._fdA||assemble(); if(a.json) copy(a.json); };
document.getElementById('fdRun').onclick=async ()=>{
  const a=assemble();
  const ep=document.getElementById('fdEpisode'), st=document.getElementById('fdStatus');
  if(!a.json){ ep.textContent=a.text; return; }
  const url=document.getElementById('fdEndpoint').value.trim();
  const model=document.getElementById('fdModel').value.trim()||'fruit';
  const messages=JSON.parse(a.json).messages;
  st.textContent='generating…'; ep.textContent='';
  try{
    const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model,messages,temperature:0.8,top_p:0.9,max_tokens:800})});
    if(!r.ok){ st.textContent='HTTP '+r.status; ep.textContent=await r.text(); return; }
    const j=await r.json();
    ep.textContent=(j.choices&&j.choices[0]&&j.choices[0].message&&j.choices[0].message.content)||JSON.stringify(j,null,2);
    st.textContent='done ✓';
  }catch(e){
    st.textContent='connection error';
    ep.textContent='Request failed: '+e.message+'\n\nChecklist:\n'
      +'  • vLLM running on the Spark (:8000) with --enable-lora\n'
      +'  • endpoint reachable from this machine (default http://spark-2e6c:8000)\n'
      +'  • vLLM started with --allowed-origins \'["*"]\' (CORS)';
  }
};
document.getElementById('fdExample').onclick=()=>{
  sel.clear();
  sel.set('Peach',{gender:'Female',stage:'prime',role:'Yoga instructor',disp:'Hot and arrogant',char:'Temptress / Seducer',note:'Runs the retreat; everyone wants her approval'});
  sel.set('Banana',{gender:'Male',stage:'prime',role:'Gym bro / trainer',disp:'Disciplined; secretly rich',char:'The Golden child',note:'Hides old family money'});
  sel.set('Watermelon',{gender:'Male',stage:'ripening',role:'Retired pro athlete',disp:'Haunted by the past',char:'The Underdog',note:'"Fat guy" who was secretly a hidden pro athlete'});
  document.getElementById('fdContext').value='A luxury wellness retreat in the hills during the annual charity gala; old rivalries resurface and a paternity secret is about to detonate.';
  renderGrid(); renderCast(); document.getElementById('fdGen').click();
};

// datalists
const dl=document.createElement('div');
dl.innerHTML=`<datalist id="fdRoles">${dataOpts(ROLES)}</datalist><datalist id="fdDisp">${dataOpts(DISPOSITIONS)}</datalist><datalist id="fdChars">${dataOpts(CHARACTERS)}</datalist>`;
document.body.appendChild(dl);
renderGrid(); renderCast();
</script>
"""

html = BODY
html = html.replace("__FRUITS__", json.dumps(DATA, ensure_ascii=False))
html = html.replace("__ROLES__", json.dumps(ROLES))
html = html.replace("__DISPOSITIONS__", json.dumps(DISPOSITIONS))
html = html.replace("__CHARACTERS__", json.dumps(CHARACTERS))
html = html.replace("__SYSPROMPT__", json.dumps(SYS))

doc = ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
       "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
       "<title>Fruit Drama — Story Builder</title></head><body>" + html + "</body></html>")

open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8").write(doc)
print("wrote", os.path.join(OUT_DIR, "index.html"))
print("fruits embedded:", len(DATA), "| doc bytes:", len(doc))
