from __future__ import annotations
import os, sys, json, subprocess, shutil, zipfile, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
from src.data_generation.generator import generate_samples, write_jsonl, read_jsonl
from src.data_generation.exporters import export_sft, export_dpo, merge_raw_with_overrides
from src.character.cognitive_state import CharacterState
from src.character.event_model import PRESET_COMPLEX_EVENTS, PLAYER_TEXT_BY_EVENT, compose_effect
from src.character.policy import choose_action, generate_rule_dialogue
from src.memory.weighted_memory import WeightedMemoryStore, MemoryItem

DATA_RAW = ROOT / 'data/raw/generated_events.jsonl'
DATA_OVERRIDES = ROOT / 'data/processed/reviewed_overrides.jsonl'
FINAL_SFT = ROOT / 'data/export/final_train_sft.jsonl'
FINAL_DPO = ROOT / 'data/export/final_train_dpo.jsonl'
FINAL_RAW = ROOT / 'data/export/final_train_raw.jsonl'
REMOTE_CODE_PACKAGE = ROOT / 'outputs/remote/SoulNPC_code_package.zip'
REMOTE_ASSET_PACKAGE = ROOT / 'outputs/remote/SoulNPC_assets_package.zip'
TRAIN_CFG = ROOT / 'configs/default_training_config.json'
DEFAULT_REMOTE_CODE_DIR = '/root/SoulNPC-Agent'
DEFAULT_REMOTE_ASSET_DIR = '/root/autodl-tmp/SoulNPC-Agent-assets'


REMOTE_PYTHON_BOOTSTRAP = '''
find_python() {
    for p in \
        /root/miniconda3/bin/python \
        /root/miniconda3/envs/*/bin/python \
        /opt/conda/bin/python \
        /usr/local/bin/python3 \
        /usr/bin/python3
    do
        if [ -x "$p" ]; then echo "$p"; return 0; fi
    done
    command -v python3 2>/dev/null || command -v python 2>/dev/null || true
}
PYTHON_BIN=$(find_python)
if [ -z "$PYTHON_BIN" ]; then
    echo "未找到可用 Python。已检查：/root/miniconda3/bin/python、/root/miniconda3/envs/*/bin/python、/opt/conda/bin/python、系统 python3/python。"
    echo "请在云端终端确认：ls -l /root/miniconda3/bin/python 或 conda env list。"
    exit 127
fi
export PYTHON_BIN
export PATH="$(dirname $PYTHON_BIN):$PATH"
echo "Using Python: $PYTHON_BIN"
'''

CSS = r'''
:root{--ink:#1f2937;--muted:#64748b;--line:#d8e1ee;--card:rgba(255,255,255,.82);--purple:#7C3AED;--pink:#EC5A8A;}
.gradio-container{background:linear-gradient(135deg,#f6f7ff 0%,#eef7f4 52%,#fff7ef 100%)!important;color:var(--ink)!important;font-family:Inter,'Microsoft YaHei',sans-serif!important;}
.wrap{max-width:1280px;margin:auto}.hero{padding:26px 30px;border-radius:32px;background:var(--card);box-shadow:0 24px 70px rgba(90,80,130,.14);border:1px solid rgba(255,255,255,.7);margin:18px 0 24px}.hero h1{font-size:34px;margin:0 0 8px;color:#0f172a}.hero p{color:var(--muted);font-size:16px;margin:0}.glass{padding:22px;border-radius:24px;background:rgba(255,255,255,.78);box-shadow:0 18px 48px rgba(79,70,120,.10);border:1px solid rgba(255,255,255,.75);margin:14px 0}.section-title{font-size:22px;font-weight:800;color:#111827;margin-bottom:8px}.hint{color:#64748b;font-size:14px;line-height:1.7}.field-title{color:#1f2937!important;font-weight:800;font-size:15px;margin:8px 0 6px 2px}.status-ok{background:#ecfdf5;border:1px solid #bbf7d0;color:#065f46;border-radius:18px;padding:14px}.status-warn{background:#fffbeb;border:1px solid #fde68a;color:#92400e;border-radius:18px;padding:14px}.status-err{background:#fef2f2;border:1px solid #fecaca;color:#991b1b;border-radius:18px;padding:14px}button{border-radius:16px!important;font-weight:800!important;transition:all .12s ease!important}button:hover{transform:translateY(-1px);box-shadow:0 12px 26px rgba(124,58,237,.18)!important}button:active{transform:translateY(1px) scale(.99)}textarea,input,select{background:#fff!important;color:#111827!important;border:1px solid #d9e2ef!important;border-radius:16px!important;min-height:42px!important}.tabs button{min-height:48px!important;border-radius:999px!important;overflow:visible!important;color:#334155!important}.tabs button.selected{background:linear-gradient(90deg,#7C3AED,#EC5A8A)!important;color:white!important}.label-wrap,.block-label{display:none!important}
'''


def html_title(text):
    return gr.HTML(f"<div class='field-title'>{text}</div>")


def count_jsonl(path: Path) -> int:
    if not path.exists(): return 0
    return sum(1 for line in open(path, 'r', encoding='utf-8') if line.strip())


def status(msg, kind='ok'):
    cls = {'ok':'status-ok','warn':'status-warn','err':'status-err'}.get(kind,'status-ok')
    return f"<div class='{cls}'><b>{msg}</b></div>"


def refresh_status():
    raw=count_jsonl(DATA_RAW); overrides=count_jsonl(DATA_OVERRIDES); sft=count_jsonl(FINAL_SFT); dpo=count_jsonl(FINAL_DPO); final_raw=count_jsonl(FINAL_RAW)
    rows=[]
    rows.append(f"自动原始样本：{raw} 条")
    rows.append(f"人工覆盖样本：{overrides} 条")
    rows.append(f"最终 SFT 样本：{sft} 条")
    rows.append(f"最终 DPO 样本：{dpo} 条")
    rows.append(f"最终 Baseline 原始结构样本：{final_raw} 条")
    rows.append("")
    for p in [DATA_RAW, DATA_OVERRIDES, FINAL_RAW, FINAL_SFT, FINAL_DPO, TRAIN_CFG]:
        rows.append(("存在" if p.exists() else "缺失") + f"：{p.relative_to(ROOT)}")
    return "\n".join(rows)


def generate_data(n, seed):
    samples=generate_samples(int(n), int(seed))
    write_jsonl(samples, str(DATA_RAW))
    export_sft(samples, str(FINAL_SFT)); export_dpo(samples, str(FINAL_DPO))
    return status(f"基础数据生成完成：{len(samples)} 条。已同步生成默认 SFT/DPO 文件。"), refresh_status()


def export_final():
    raw=read_jsonl(str(DATA_RAW))
    if not raw:
        return status('缺少原始样本，请先在数据工厂生成基础训练样本。','err'), refresh_status()
    overrides=read_jsonl(str(DATA_OVERRIDES))
    merged=merge_raw_with_overrides(raw, overrides)
    write_jsonl(merged, str(FINAL_RAW))
    export_sft(merged, str(FINAL_SFT)); export_dpo(merged, str(FINAL_DPO))
    return status(f"最终训练数据已生成：原始 {len(raw)} 条，人工覆盖 {len(overrides)} 条，最终 SFT {len(merged)} 条。"), refresh_status()


def train_baseline():
    ok_msg, _ = export_final()
    out_dir = ROOT / 'outputs/baseline'
    cmd=[sys.executable, str(ROOT/'scripts/train_baseline.py'), '--input', str(FINAL_RAW), '--out', str(out_dir)]
    p=subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding='utf-8')
    if p.returncode!=0:
        return status('Baseline 训练失败，请查看错误。','err') + f"\n\n{p.stderr}"
    snippets=[]
    for name in ['derived_mood_report.txt','action_report.txt']:
        fp=out_dir/name
        if fp.exists(): snippets.append(fp.read_text(encoding='utf-8')[:1800])
    return status('Baseline 训练完成。') + f"\n\n输出目录：{out_dir.relative_to(ROOT)}\n\n" + "\n\n".join(snippets)


def save_override(sample_id, reviewed_text, chosen, rejected):
    if not sample_id:
        return status('没有样本 ID，无法保存。','err')
    DATA_OVERRIDES.parent.mkdir(parents=True, exist_ok=True)
    row={'id':sample_id,'reviewed_dialogue':reviewed_text,'chosen':chosen,'rejected':rejected,'saved_at':datetime.now().isoformat(timespec='seconds')}
    with open(DATA_OVERRIDES,'a',encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False)+'\n')
    return status(f'已保存人工覆盖样本：{sample_id}')


def load_sample(idx):
    rows=read_jsonl(str(DATA_RAW))
    if not rows:
        return '', '', '', '', '', status('没有可审阅样本，请先生成数据。','warn')
    idx=max(0,min(int(idx),len(rows)-1)); s=rows[idx]
    chosen=s.get('dialogue',''); rejected=s.get('rejected_dialogue','')
    info=json.dumps({k:s[k] for k in ['id','event_name','event_primitives','player_text','derived_mood','action','importance']},ensure_ascii=False,indent=2)
    prompt=json.dumps(s.get('sft_prompt',{}),ensure_ascii=False,indent=2)
    return s['id'], info, prompt, chosen, rejected, status(f'已加载样本 {idx+1}/{len(rows)}：{s["id"]}')


def clean_data(mode, confirm):
    if confirm.strip()!='确认清理': return status('请输入“确认清理”后再执行。','warn'), refresh_status()
    targets=[]
    if mode=='仅清理人工覆盖': targets=[DATA_OVERRIDES]
    elif mode=='清理训练导出': targets=[FINAL_SFT,FINAL_DPO]
    elif mode=='清理自动生成数据': targets=[DATA_RAW]
    else: targets=[DATA_RAW,DATA_OVERRIDES,FINAL_SFT,FINAL_DPO]
    for p in targets:
        if p.exists(): p.unlink()
    return status(f'清理完成：{mode}'), refresh_status()


def save_train_config(base_model, model_source, local_model_path, hf_endpoint, train_file, out_dir, epochs, lr, r, alpha):
    cfg=json.load(open(TRAIN_CFG,'r',encoding='utf-8')) if TRAIN_CFG.exists() else {}
    source_map = {
        'ModelScope（推荐，适合 AutoDL 国内网络）': 'modelscope',
        'HuggingFace 官方/镜像': 'huggingface',
        '本地模型路径': 'local',
    }
    cfg.update({
        'base_model':base_model, 'model_source': source_map.get(model_source, 'modelscope'),
        'local_model_path': local_model_path or '', 'hf_endpoint': hf_endpoint or '',
        'train_file':train_file, 'output_dir':out_dir,
        'num_train_epochs':float(epochs), 'learning_rate':float(lr), 'lora_r':int(r), 'lora_alpha':int(alpha)
    })
    TRAIN_CFG.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    return status('LoRA 训练配置已保存。'), make_commands()


def make_commands():
    local = "python3 -m pip install -r requirements_lora_optional.txt\npython3 scripts/train_lora_sft.py --config configs/default_training_config.json"
    autodl = (
        f"cd {DEFAULT_REMOTE_CODE_DIR}\n"
        "# 自动查找 /root/miniconda3 或系统 Python\n"
        f"$PYTHON_BIN scripts/train_lora_sft.py --config {DEFAULT_REMOTE_ASSET_DIR}/configs/default_training_config.json"
    )
    return f"本地训练命令：\n{local}\n\nAutoDL 训练命令：\n{autodl}"


def infer(event_name, player_text):
    state=CharacterState()
    primitives=PRESET_COMPLEX_EVENTS.get(event_name, {})
    eff=compose_effect(primitives)
    before=state.to_dict()
    state.affect.update(eff['affect'], momentum=0.65); state.relationship.update(eff['relationship'])
    after=state.to_dict()
    action=choose_action(after, primitives)
    dialogue=generate_rule_dialogue(action, after)
    mem=MemoryItem(text=f"事件：{event_name}；玩家说：{player_text}", event=event_name, importance=0.7, emotional_salience=0.6, relation_impact=0.6)
    return dialogue, json.dumps(after, ensure_ascii=False, indent=2), f"行为意图：{action}\n新增记忆：{mem.text}"




def _remote_require_paramiko():
    try:
        import paramiko
        return paramiko, None
    except Exception as exc:
        return None, f"缺少 paramiko：{exc}\n请先运行：pip install paramiko"


def _connect_ssh(host, port, username, password):
    paramiko, err = _remote_require_paramiko()
    if err:
        raise RuntimeError(err)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=str(host).strip(), port=int(port), username=str(username).strip(), password=password, timeout=20)
    return client


def _ssh_exec(host, port, username, password, command, timeout=600):
    client = _connect_ssh(host, port, username, password)
    try:
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        code = stdout.channel.recv_exit_status()
        return code, out, err
    finally:
        client.close()


def autodl_check_env(host, port, username, password):
    if not password:
        return status('请在密码框中输入 AutoDL SSH 密码。本工具不会保存密码。', 'warn')
    cmd = f"""
echo '=== GPU ==='; nvidia-smi || true
echo '=== Conda / Miniconda ==='; (ls -ld /root/miniconda3 2>/dev/null || true); (command -v conda 2>/dev/null || true); (conda env list 2>/dev/null || true)
echo '=== Python discovery ==='
{REMOTE_PYTHON_BOOTSTRAP}
echo '=== Python ==='; $PYTHON_BIN --version || true
echo '=== Pip ==='; $PYTHON_BIN -m pip --version || true
echo '=== Disk ==='; df -h | head -20
echo '=== PWD ==='; pwd
"""
    try:
        code, out, err = _ssh_exec(host, port, username, password, cmd, timeout=60)
        kind = 'ok' if code == 0 else 'err'
        return status('云端环境检查完成。' if code == 0 else '云端环境检查失败。', kind) + f"\n\n{out}\n{err}"
    except Exception as exc:
        return status('连接 AutoDL 失败。', 'err') + f"\n\n{exc}"



def _write_remote_training_config(asset_dir: str):
    """生成云端专用训练配置：代码放 /root/SoulNPC-Agent，数据/输出放 asset_dir。"""
    export_final()
    cfg = json.load(open(TRAIN_CFG, 'r', encoding='utf-8')) if TRAIN_CFG.exists() else {}
    asset_dir = str(asset_dir).rstrip('/') or DEFAULT_REMOTE_ASSET_DIR
    cfg['train_file'] = f'{asset_dir}/data/export/final_train_sft.jsonl'
    cfg['output_dir'] = f'{asset_dir}/outputs/lora_sft'
    cfg['dpo_train_file'] = f'{asset_dir}/data/export/final_train_dpo.jsonl'
    cfg.setdefault('model_source', 'modelscope')
    cfg.setdefault('local_model_path', '')
    cfg.setdefault('hf_endpoint', '')
    if not cfg.get('model_cache_dir'):
        cfg['model_cache_dir'] = f'{asset_dir}/cache/modelscope' if cfg.get('model_source', 'modelscope') == 'modelscope' else f'{asset_dir}/cache/huggingface'
    out_cfg_dir = ROOT / 'outputs/remote/configs'
    out_cfg_dir.mkdir(parents=True, exist_ok=True)
    out_cfg = out_cfg_dir / 'default_training_config.remote.json'
    out_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    return out_cfg


def make_remote_package(asset_dir: str = DEFAULT_REMOTE_ASSET_DIR):
    """生成两个包：代码包进入 /root/SoulNPC-Agent；资产包进入 /root/autodl-tmp/SoulNPC-Agent-assets。"""
    export_final()
    REMOTE_CODE_PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    for fp in [REMOTE_CODE_PACKAGE, REMOTE_ASSET_PACKAGE]:
        if fp.exists():
            fp.unlink()

    exclude_dirs = {'myagent', '.venv', 'venv', '__pycache__', '.git'}
    code_top = {'app', 'src', 'scripts'}
    code_files = {'requirements.txt', 'requirements_lora_optional.txt', 'README.md'}

    with zipfile.ZipFile(REMOTE_CODE_PACKAGE, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in ROOT.rglob('*'):
            rel = path.relative_to(ROOT).as_posix()
            if any(part in exclude_dirs for part in path.parts):
                continue
            if path.is_file() and (rel.split('/')[0] in code_top or rel in code_files):
                zf.write(path, rel)

    remote_cfg = _write_remote_training_config(asset_dir)
    with zipfile.ZipFile(REMOTE_ASSET_PACKAGE, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in [
            (FINAL_SFT, 'data/export/final_train_sft.jsonl'),
            (FINAL_DPO, 'data/export/final_train_dpo.jsonl'),
            (FINAL_RAW, 'data/export/final_train_raw.jsonl'),
            (remote_cfg, 'configs/default_training_config.json'),
        ]:
            if Path(src).exists():
                zf.write(src, arc)

    code_mb = REMOTE_CODE_PACKAGE.stat().st_size / 1024 / 1024
    asset_mb = REMOTE_ASSET_PACKAGE.stat().st_size / 1024 / 1024
    msg = (
        f"云端训练包已生成。\n"
        f"代码包：{REMOTE_CODE_PACKAGE.relative_to(ROOT)}，{code_mb:.2f} MB\n"
        f"资产包：{REMOTE_ASSET_PACKAGE.relative_to(ROOT)}，{asset_mb:.2f} MB\n\n"
        f"远程路径策略：\n"
        f"代码目录：{DEFAULT_REMOTE_CODE_DIR}\n"
        f"资产/输出目录：{asset_dir}\n"
    )
    return status('云端训练包已生成。') + "\n\n" + msg, str(REMOTE_CODE_PACKAGE)


def autodl_upload_project(host, port, username, password, code_dir, asset_dir):
    if not password:
        return status('请在密码框中输入 AutoDL SSH 密码。本工具不会保存密码。', 'warn')
    try:
        code_dir = str(code_dir).rstrip('/') or DEFAULT_REMOTE_CODE_DIR
        asset_dir = str(asset_dir).rstrip('/') or DEFAULT_REMOTE_ASSET_DIR
        _, _ = make_remote_package(asset_dir)
        remote_code_zip = '/root/autodl-tmp/SoulNPC_code_package.zip'
        remote_asset_zip = '/root/autodl-tmp/SoulNPC_assets_package.zip'
        _ssh_exec(host, port, username, password, 'mkdir -p /root/autodl-tmp', timeout=60)
        client = _connect_ssh(host, port, username, password)
        try:
            sftp = client.open_sftp()
            sftp.put(str(REMOTE_CODE_PACKAGE), remote_code_zip)
            sftp.put(str(REMOTE_ASSET_PACKAGE), remote_asset_zip)
            sftp.close()
        finally:
            client.close()
        cmd = f"""
set -e
rm -rf {code_dir}
mkdir -p {code_dir}
mkdir -p {asset_dir}
{REMOTE_PYTHON_BOOTSTRAP}
$PYTHON_BIN - <<'REMOTE_PY'
import zipfile
zipfile.ZipFile('{remote_code_zip}').extractall('{code_dir}')
zipfile.ZipFile('{remote_asset_zip}').extractall('{asset_dir}')
print('code extracted to {code_dir}')
print('assets extracted to {asset_dir}')
REMOTE_PY
mkdir -p {asset_dir}/outputs {asset_dir}/logs {asset_dir}/cache
echo '=== code dir ==='
ls -la {code_dir} | head
echo '=== asset dir ==='
find {asset_dir} -maxdepth 3 -type f | head -30
"""
        code, out, err = _ssh_exec(host, port, username, password, cmd, timeout=300)
        if code != 0:
            return status('上传后解压失败。', 'err') + f"\n\n{out}\n{err}"
        return status('代码与资产已按指定路径上传到 AutoDL。') + f"\n\n代码目录：{code_dir}\n资产/输出目录：{asset_dir}\n\n{out}"
    except Exception as exc:
        return status('上传项目到 AutoDL 失败。', 'err') + f"\n\n{exc}"


def autodl_install_deps(host, port, username, password, code_dir):
    code_dir = str(code_dir).rstrip('/') or DEFAULT_REMOTE_CODE_DIR
    cmd = f'''
set -e
cd {code_dir}
{REMOTE_PYTHON_BOOTSTRAP}
echo "=== install pip / wheel tools ==="
$PYTHON_BIN -m pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple || $PYTHON_BIN -m pip install -U pip setuptools wheel
echo "=== install lora requirements ==="
$PYTHON_BIN -m pip install -r requirements_lora_optional.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || $PYTHON_BIN -m pip install -r requirements_lora_optional.txt
echo "=== ensure ModelScope / HuggingFace Hub ==="
$PYTHON_BIN -m pip install -U modelscope huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple || $PYTHON_BIN -m pip install -U modelscope huggingface_hub
echo "=== verify core packages ==="
$PYTHON_BIN - <<'VERIFY_PY'
import importlib
mods = ['torch', 'transformers', 'datasets', 'peft', 'modelscope', 'huggingface_hub']
for m in mods:
    importlib.import_module(m)
    print(m + ' ok')
VERIFY_PY
'''
    try:
        code, out, err = _ssh_exec(host, port, username, password, cmd, timeout=1800)
        return status('云端训练依赖安装完成。' if code == 0 else '云端训练依赖安装失败。', 'ok' if code == 0 else 'err') + f"\n\n{out[-6000:]}\n{err[-6000:]}"
    except Exception as exc:
        return status('安装依赖失败。', 'err') + f"\n\n{exc}"


def autodl_install_deps(host, port, username, password, code_dir):
    code_dir = str(code_dir).rstrip('/') or DEFAULT_REMOTE_CODE_DIR
    cmd = f'''
set -e
cd {code_dir}
{REMOTE_PYTHON_BOOTSTRAP}
echo "=== install pip / wheel tools ==="
$PYTHON_BIN -m pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple || $PYTHON_BIN -m pip install -U pip setuptools wheel
echo "=== install lora requirements ==="
$PYTHON_BIN -m pip install -r requirements_lora_optional.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || $PYTHON_BIN -m pip install -r requirements_lora_optional.txt
echo "=== ensure ModelScope / HuggingFace Hub ==="
$PYTHON_BIN -m pip install -U modelscope huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple || $PYTHON_BIN -m pip install -U modelscope huggingface_hub
echo "=== verify core packages ==="
$PYTHON_BIN - <<'VERIFY_PY'
import importlib
mods = ['torch', 'transformers', 'datasets', 'peft', 'modelscope', 'huggingface_hub']
for m in mods:
    importlib.import_module(m)
    print(m + ' ok')
VERIFY_PY
'''
    try:
        code, out, err = _ssh_exec(host, port, username, password, cmd, timeout=1800)
        return status('云端训练依赖安装完成。' if code == 0 else '云端训练依赖安装失败。', 'ok' if code == 0 else 'err') + f"\n\n{out[-6000:]}\n{err[-6000:]}"
    except Exception as exc:
        return status('安装依赖失败。', 'err') + f"\n\n{exc}"



def _normal_model_source(model_source):
    s = str(model_source or '').lower()
    if 'modelscope' in s or '魔搭' in s:
        return 'modelscope'
    if 'huggingface' in s or '镜像' in s:
        return 'huggingface'
    if '本地' in s or 'local' in s:
        return 'local'
    return 'modelscope'


def _safe_model_dir_name(model_id: str) -> str:
    return str(model_id or 'base-model').replace('/', '__').replace(' ', '_')


def autodl_check_base_model(host, port, username, password, asset_dir, local_model_dir):
    if not password:
        return status('请在密码框中输入 AutoDL SSH 密码。本工具不会保存密码。', 'warn')
    asset_dir = str(asset_dir).rstrip('/') or DEFAULT_REMOTE_ASSET_DIR
    local_model_dir = str(local_model_dir).rstrip('/') or '/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct'
    cmd = f'''
set -e
{REMOTE_PYTHON_BOOTSTRAP}
echo '=== model directory ==='
echo '{local_model_dir}'
if [ -d '{local_model_dir}' ]; then
    echo '模型目录存在。'
    find '{local_model_dir}' -maxdepth 2 -type f | sed -n '1,40p'
    if [ -f '{local_model_dir}/config.json' ]; then
        echo 'config.json 存在，模型目录基本可用。'
    else
        echo '警告：未找到 config.json，这个目录可能不是 Transformers 可直接加载的模型目录。'
    fi
else
    echo '模型目录不存在。请先点击“下载/准备基础模型”。'
fi

echo '=== remote config ==='
cat '{asset_dir}/configs/default_training_config.json' 2>/dev/null || echo '远程训练配置不存在，请先上传项目与数据。'
'''
    try:
        code, out, err = _ssh_exec(host, port, username, password, cmd, timeout=120)
        return status('基础模型检查完成。' if code == 0 else '基础模型检查失败。', 'ok' if code == 0 else 'err') + f"\n\n{out}\n{err}"
    except Exception as exc:
        return status('检查基础模型失败。', 'err') + f"\n\n{exc}"


def autodl_prepare_base_model(host, port, username, password, code_dir, asset_dir, model_id, model_source, local_model_dir, hf_endpoint):
    if not password:
        return status('请在密码框中输入 AutoDL SSH 密码。本工具不会保存密码。', 'warn')
    code_dir = str(code_dir).rstrip('/') or DEFAULT_REMOTE_CODE_DIR
    asset_dir = str(asset_dir).rstrip('/') or DEFAULT_REMOTE_ASSET_DIR
    model_id = str(model_id or 'Qwen/Qwen2.5-1.5B-Instruct').strip()
    source = _normal_model_source(model_source)
    if not str(local_model_dir or '').strip():
        local_model_dir = f"/root/autodl-tmp/models/{_safe_model_dir_name(model_id)}"
    local_model_dir = str(local_model_dir).rstrip('/')
    hf_endpoint = str(hf_endpoint or '').strip()
    import json as _json
    env_block = "\n".join([
        f"export SOUL_MODEL_ID={_json.dumps(model_id)}",
        f"export SOUL_MODEL_SOURCE={_json.dumps(source)}",
        f"export SOUL_LOCAL_MODEL_DIR={_json.dumps(local_model_dir)}",
        f"export SOUL_ASSET_DIR={_json.dumps(asset_dir)}",
        f"export SOUL_HF_ENDPOINT={_json.dumps(hf_endpoint)}",
    ])
    cmd = f'''
set -e
mkdir -p {asset_dir}/cache/modelscope {asset_dir}/cache/huggingface /root/autodl-tmp/models
cd {code_dir}
{REMOTE_PYTHON_BOOTSTRAP}
{env_block}
export HF_HOME={asset_dir}/cache/huggingface
export TRANSFORMERS_CACHE={asset_dir}/cache/huggingface
export MODELSCOPE_CACHE={asset_dir}/cache/modelscope
if [ -n "$SOUL_HF_ENDPOINT" ]; then
    export HF_ENDPOINT="$SOUL_HF_ENDPOINT"
fi
if [ "$SOUL_MODEL_SOURCE" = "modelscope" ]; then
    echo "=== verify modelscope before download ==="
    if ! $PYTHON_BIN - <<'CHECK_MODELSCOPE'; then
import modelscope
print('modelscope import ok')
CHECK_MODELSCOPE
        echo "ModelScope 未安装，正在自动安装到当前 Python 环境..."
        $PYTHON_BIN -m pip install -U modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple || $PYTHON_BIN -m pip install -U modelscope
        $PYTHON_BIN - <<'CHECK_MODELSCOPE_AFTER'
import modelscope
print('modelscope import ok after install')
CHECK_MODELSCOPE_AFTER
    fi
fi
$PYTHON_BIN - <<'REMOTE_PY'
import os, json
from pathlib import Path
model_id = os.environ['SOUL_MODEL_ID']
source = os.environ['SOUL_MODEL_SOURCE']
local_dir = Path(os.environ['SOUL_LOCAL_MODEL_DIR']).expanduser()
asset_dir = Path(os.environ['SOUL_ASSET_DIR']).expanduser()
hf_endpoint = os.environ.get('SOUL_HF_ENDPOINT','')
asset_dir.mkdir(parents=True, exist_ok=True)
local_dir.parent.mkdir(parents=True, exist_ok=True)
print('[SoulNPC] 模型ID: ' + str(model_id), flush=True)
print('[SoulNPC] 下载来源: ' + str(source), flush=True)
print('[SoulNPC] 目标本地目录: ' + str(local_dir), flush=True)

final_dir = local_dir
if source == 'local':
    if not local_dir.exists():
        raise FileNotFoundError('本地模型目录不存在: ' + str(local_dir))
    print('[SoulNPC] 仅检查本地模型目录。', flush=True)
elif source == 'modelscope':
    try:
        from modelscope import snapshot_download
    except Exception as exc:
        raise RuntimeError('缺少 modelscope，请先安装云端训练依赖。') from exc
    cache_dir = str(asset_dir / 'cache' / 'modelscope')
    downloaded = Path(snapshot_download(model_id, cache_dir=cache_dir))
    print('[SoulNPC] ModelScope 下载/缓存目录: ' + str(downloaded), flush=True)
    if not local_dir.exists():
        try:
            local_dir.symlink_to(downloaded, target_is_directory=True)
            final_dir = local_dir
            print('[SoulNPC] 已创建软链接: ' + str(local_dir) + ' -> ' + str(downloaded), flush=True)
        except Exception as exc:
            final_dir = downloaded
            print('[SoulNPC] 软链接失败，直接使用 ModelScope 缓存目录: ' + str(exc), flush=True)
    else:
        final_dir = local_dir
elif source == 'huggingface':
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError('缺少 huggingface_hub，请先安装云端训练依赖。') from exc
    if hf_endpoint:
        os.environ['HF_ENDPOINT'] = hf_endpoint
        print('[SoulNPC] HF_ENDPOINT=' + str(hf_endpoint), flush=True)
    downloaded = Path(snapshot_download(repo_id=model_id, local_dir=str(local_dir), local_dir_use_symlinks=False))
    final_dir = downloaded
else:
    raise ValueError('未知模型来源: ' + str(source))

if not (final_dir / 'config.json').exists():
    raise FileNotFoundError('未找到 config.json，模型目录不可用: ' + str(final_dir))
print('[SoulNPC] 可用模型目录: ' + str(final_dir), flush=True)

cfg_path = asset_dir / 'configs' / 'default_training_config.json'
if not cfg_path.exists():
    raise FileNotFoundError('远程训练配置不存在，请先上传项目与数据: ' + str(cfg_path))
cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
cfg['model_source'] = 'local'
cfg['base_model'] = str(final_dir)
cfg['local_model_path'] = str(final_dir)
cfg['model_cache_dir'] = str(asset_dir / 'cache')
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
print('[SoulNPC] 已把远程训练配置切换为本地模型路径。', flush=True)
print(json.dumps(dict(local_model_path=str(final_dir), config_path=str(cfg_path)), ensure_ascii=False, indent=2), flush=True)
REMOTE_PY
'''
    try:
        code, out, err = _ssh_exec(host, port, username, password, cmd, timeout=7200)
        return status('基础模型已准备完成，远程训练将从本地模型目录加载。' if code == 0 else '基础模型准备失败。', 'ok' if code == 0 else 'err') + f"\n\n{out[-10000:]}\n{err[-10000:]}"
    except Exception as exc:
        return status('准备基础模型失败。', 'err') + f"\n\n{exc}"

def autodl_start_sft(host, port, username, password, code_dir, asset_dir):
    code_dir = str(code_dir).rstrip('/') or DEFAULT_REMOTE_CODE_DIR
    asset_dir = str(asset_dir).rstrip('/') or DEFAULT_REMOTE_ASSET_DIR
    config_path = f'{asset_dir}/configs/default_training_config.json'
    log_path = f'{asset_dir}/outputs/sft_train.log'
    run_script = '/root/autodl-tmp/soulnpc_sft_run.sh'
    cmd = f'''
set -e
mkdir -p {asset_dir}/outputs /root/autodl-tmp
cat > {run_script} <<'REMOTE_RUN'
set -e
cd {code_dir}
{REMOTE_PYTHON_BOOTSTRAP}
export HF_HOME={asset_dir}/cache/huggingface
export TRANSFORMERS_CACHE={asset_dir}/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export MODELSCOPE_CACHE={asset_dir}/cache/modelscope
mkdir -p {asset_dir}/cache/huggingface {asset_dir}/cache/modelscope {asset_dir}/outputs
$PYTHON_BIN scripts/train_lora_sft.py --config {config_path} 2>&1 | tee {log_path}
REMOTE_RUN
chmod +x {run_script}
PID_FILE={asset_dir}/outputs/soulnpc_sft.pid
LAUNCHER_LOG={asset_dir}/outputs/sft_launcher.log
if command -v tmux >/dev/null 2>&1; then
    echo "Using tmux backend."
    (tmux kill-session -t soulnpc_sft 2>/dev/null || true)
    tmux new -d -s soulnpc_sft "bash {run_script}"
    tmux ls
else
    echo "tmux not found. Falling back to nohup backend."
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
        if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Stopping previous nohup training process: $OLD_PID"
            kill "$OLD_PID" 2>/dev/null || true
        fi
    fi
    nohup bash {run_script} > "$LAUNCHER_LOG" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started with nohup PID: $(cat "$PID_FILE")"
    echo "Launcher log: $LAUNCHER_LOG"
fi
'''
    try:
        code, out, err = _ssh_exec(host, port, username, password, cmd, timeout=60)
        return status('SFT LoRA 训练已在 AutoDL 后台启动。' if code == 0 else 'SFT LoRA 启动失败。', 'ok' if code == 0 else 'err') + f"\n\n日志文件：{log_path}\n输出目录：{asset_dir}/outputs/lora_sft\n运行脚本：{run_script}\n\n{out}\n{err}"
    except Exception as exc:
        return status('启动 SFT 训练失败。', 'err') + f"\n\n{exc}"


def autodl_read_log(host, port, username, password, asset_dir, lines):
    asset_dir = str(asset_dir).rstrip('/') or DEFAULT_REMOTE_ASSET_DIR
    n = int(lines or 120)
    cmd = f'''
echo '=== process backend ==='
if command -v tmux >/dev/null 2>&1; then
    tmux ls 2>/dev/null || echo 'tmux installed, but no active session.'
else
    echo 'tmux not installed. Using nohup fallback.'
fi
PID_FILE={asset_dir}/outputs/soulnpc_sft.pid
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null || true)
    echo "nohup pid: $PID"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo 'nohup process: running'
    else
        echo 'nohup process: not running or finished'
    fi
fi
echo '=== launcher log ==='
tail -n 40 {asset_dir}/outputs/sft_launcher.log 2>/dev/null || echo '暂无 launcher 日志'
echo '=== train log tail ==='
tail -n {n} {asset_dir}/outputs/sft_train.log 2>/dev/null || echo '暂无训练日志'
'''
    try:
        code, out, err = _ssh_exec(host, port, username, password, cmd, timeout=60)
        return out + ('\n' + err if err else '')
    except Exception as exc:
        return f"读取日志失败：{exc}"


def autodl_check_training(host, port, username, password, asset_dir):
    asset_dir = str(asset_dir).rstrip('/') or DEFAULT_REMOTE_ASSET_DIR
    cmd = f'''
RUNNING=0
if command -v tmux >/dev/null 2>&1 && tmux has-session -t soulnpc_sft 2>/dev/null; then
    echo '训练状态：运行中（tmux）'
    RUNNING=1
else
    PID_FILE={asset_dir}/outputs/soulnpc_sft.pid
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE" 2>/dev/null || true)
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo "训练状态：运行中（nohup，PID=$PID）"
            RUNNING=1
        else
            echo '训练状态：未运行或已结束（nohup pid 不存在/已结束）'
        fi
    else
        echo '训练状态：未运行或已结束'
    fi
fi
echo '=== adapter files ==='
find {asset_dir}/outputs/lora_sft -maxdepth 2 -type f 2>/dev/null | head -30 || true
echo '=== latest log ==='
tail -n 80 {asset_dir}/outputs/sft_train.log 2>/dev/null || true
'''
    try:
        code, out, err = _ssh_exec(host, port, username, password, cmd, timeout=60)
        return out + ('\n' + err if err else '')
    except Exception as exc:
        return f"检查训练状态失败：{exc}"


def autodl_infer(host, port, username, password, code_dir, asset_dir, prompt):
    code_dir = str(code_dir).rstrip('/') or DEFAULT_REMOTE_CODE_DIR
    asset_dir = str(asset_dir).rstrip('/') or DEFAULT_REMOTE_ASSET_DIR
    prompt = str(prompt).replace('"', '\\"')
    config_path = f'{asset_dir}/configs/default_training_config.json'
    cmd = f"""
set -e
cd {code_dir}
{REMOTE_PYTHON_BOOTSTRAP}
export HF_HOME={asset_dir}/cache/huggingface
export TRANSFORMERS_CACHE={asset_dir}/cache/huggingface
export MODELSCOPE_CACHE={asset_dir}/cache/modelscope
$PYTHON_BIN scripts/infer_lora.py --config {config_path} --prompt "{prompt}"
"""
    try:
        code, out, err = _ssh_exec(host, port, username, password, cmd, timeout=600)
        return status('云端 LoRA 推理完成。' if code == 0 else '云端 LoRA 推理失败。', 'ok' if code == 0 else 'err') + f"\n\n{out}\n{err}"
    except Exception as exc:
        return status('云端推理失败。', 'err') + f"\n\n{exc}"


def autodl_start_server(host, port, username, password, code_dir, asset_dir, service_port):
    code_dir = str(code_dir).rstrip('/') or DEFAULT_REMOTE_CODE_DIR
    asset_dir = str(asset_dir).rstrip('/') or DEFAULT_REMOTE_ASSET_DIR
    sp = int(service_port or 6006)
    config_path = f'{asset_dir}/configs/default_training_config.json'
    log_path = f'{asset_dir}/outputs/serve_lora.log'
    run_script = '/root/autodl-tmp/soulnpc_serve_run.sh'
    cmd = f'''
set -e
mkdir -p {asset_dir}/outputs /root/autodl-tmp
cat > {run_script} <<'REMOTE_RUN'
set -e
cd {code_dir}
{REMOTE_PYTHON_BOOTSTRAP}
export HF_HOME={asset_dir}/cache/huggingface
export TRANSFORMERS_CACHE={asset_dir}/cache/huggingface
export MODELSCOPE_CACHE={asset_dir}/cache/modelscope
$PYTHON_BIN scripts/serve_lora.py --config {config_path} --port {sp} 2>&1 | tee {log_path}
REMOTE_RUN
chmod +x {run_script}
PID_FILE={asset_dir}/outputs/soulnpc_serve.pid
LAUNCHER_LOG={asset_dir}/outputs/serve_launcher.log
if command -v tmux >/dev/null 2>&1; then
    echo "Using tmux backend for serving."
    (tmux kill-session -t soulnpc_serve 2>/dev/null || true)
    tmux new -d -s soulnpc_serve "bash {run_script}"
    tmux ls
else
    echo "tmux not found. Falling back to nohup backend for serving."
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
        if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Stopping previous serve process: $OLD_PID"
            kill "$OLD_PID" 2>/dev/null || true
        fi
    fi
    nohup bash {run_script} > "$LAUNCHER_LOG" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started serve with nohup PID: $(cat "$PID_FILE")"
    echo "Launcher log: $LAUNCHER_LOG"
fi
'''
    try:
        code, out, err = _ssh_exec(host, port, username, password, cmd, timeout=60)
        return status('云端推理服务已启动。' if code == 0 else '云端推理服务启动失败。', 'ok' if code == 0 else 'err') + f"\n\n监听端口：{sp}\n日志文件：{log_path}\n运行脚本：{run_script}\nAutoDL 6006/6008 映射地址可直接访问。\n\n{out}\n{err}"
    except Exception as exc:
        return status('启动推理服务失败。', 'err') + f"\n\n{exc}"



def estimate_training_time(sample_count=None, epochs=None, model_size='1.5B', gpu_name='RTX 5090'):
    """基于样本数、轮数、模型规模给出粗略训练时长预估。不是精确计时，只用于设置 AutoDL 关机缓冲。"""
    try:
        if sample_count is None or int(sample_count) <= 0:
            sample_count = count_jsonl(FINAL_SFT)
        sample_count = max(1, int(sample_count))
    except Exception:
        sample_count = count_jsonl(FINAL_SFT) or 1000
    try:
        epochs = max(1, float(epochs or 1))
    except Exception:
        epochs = 1.0
    model_size = str(model_size or '1.5B')
    # 经验估算：5090 上 1.5B QLoRA 单样本约 0.045-0.09 秒，实际受 max_length、batch、下载和首次编译影响很大。
    if '0.5' in model_size or '0.6' in model_size:
        sec_per_sample = 0.025
    elif '1.5' in model_size or '1.7' in model_size:
        sec_per_sample = 0.065
    elif '3' in model_size:
        sec_per_sample = 0.12
    elif '7' in model_size:
        sec_per_sample = 0.32
    else:
        sec_per_sample = 0.08
    train_seconds = sample_count * epochs * sec_per_sample
    # 首次下载模型、安装依赖、tokenizer map、保存 adapter 的固定缓冲。首次训练可能更多。
    fixed_buffer_min = 25
    train_min = max(2, train_seconds / 60)
    total_min = train_min + fixed_buffer_min
    shutdown_min = int(total_min * 1.35 + 20)
    return (
        status('训练时长预估已生成。') +
        f"\n\n样本数：{sample_count}\n训练轮数：{epochs}\n模型规模：{model_size}\nGPU：{gpu_name}\n\n"
        f"预计纯训练时间：约 {train_min:.1f} 分钟\n"
        f"含模型下载/预处理/保存缓冲：约 {total_min:.1f} 分钟\n"
        f"建议 AutoDL 关机时间：至少 {shutdown_min} 分钟后\n\n"
        "说明：这是保守估算。第一次运行需要下载模型和安装依赖，可能明显更慢；第二次以后通常更接近纯训练时间。"
    )

def build_app():
    with gr.Blocks(css=CSS, title='SoulNPC 中文训练工作台 v1.8.1 AutoDL 模型下载修复版') as demo:
        gr.HTML("<div class='wrap'><div class='hero'><h1> SoulNPC 中文训练工作台 v1.8.1 AutoDL 模型下载修复版</h1><p>面向可信游戏角色的轻量认知-情感 Agent：模型下载、LoRA 训练、日志监控与云端推理一体化。</p></div></div>")
        with gr.Tabs():
            with gr.Tab('0. 控制台'):
                gr.HTML("<div class='glass'><div class='section-title'>项目总览</div><div class='hint'>这里显示自动样本、人工覆盖、最终训练文件和训练配置状态。</div></div>")
                btn=gr.Button('刷新项目状态', variant='primary')
                box=gr.Textbox(lines=14, show_label=False)
                btn.click(refresh_status, outputs=box)
            with gr.Tab('1. 数据工厂'):
                gr.HTML("<div class='glass'><div class='section-title'>自动生成训练数据</div><div class='hint'>没有人工审阅也能训练；人工审阅只是覆盖增强层。</div></div>")
                with gr.Row():
                    n=gr.Number(value=1000, label='基础训练样本数')
                    seed=gr.Number(value=42, label='复现实验编号')
                gen=gr.Button('一键生成基础训练样本', variant='primary')
                gen_status=gr.HTML(); stat=gr.Textbox(lines=8, show_label=False)
                gen.click(generate_data, inputs=[n,seed], outputs=[gen_status, stat])
                gr.HTML("<div class='glass'><div class='section-title'>缓存清理</div><div class='hint'>输入“确认清理”后执行，防止误删。</div></div>")
                clean_mode=gr.Dropdown(['仅清理人工覆盖','清理训练导出','清理自动生成数据','全量清理'], value='仅清理人工覆盖', label='清理范围')
                confirm=gr.Textbox(value='', placeholder='输入：确认清理', label='安全确认')
                clean_btn=gr.Button('执行清理')
                clean_out=gr.HTML(); clean_stat=gr.Textbox(lines=6, show_label=False)
                clean_btn.click(clean_data, inputs=[clean_mode,confirm], outputs=[clean_out,clean_stat])
            with gr.Tab('2. 事件与记忆机制'):
                gr.HTML("<div class='glass'><div class='section-title'>可训练的事件组合与加权记忆</div><div class='hint'>复杂事件由小事件原语和权重组合。记忆带重要度、情绪显著性和关系影响权重，低权重记忆可随机遗忘。</div></div>")
                gr.JSON(value=PRESET_COMPLEX_EVENTS, label='复杂事件 = 抽象事件原语 + 权重')
                mem_info=gr.Markdown("""
### 记忆权重公式（当前规则版）
`memory_score = 0.45*importance + 0.25*emotional_salience + 0.20*relation_impact + 0.10*recency`

后续可训练为 Memory Scorer：输入记忆文本、事件类型、关系变化、后续对话质量，学习哪些记忆应该长期保留。
                """)
            with gr.Tab('3. 审阅覆盖'):
                gr.HTML("<div class='glass'><div class='section-title'>人工偏好覆盖</div><div class='hint'>不审阅也能训练；审阅后会覆盖同 ID 原样本，提高 SFT/DPO 质量。</div></div>")
                idx=gr.Number(value=0, label='样本序号')
                load=gr.Button('加载样本')
                sid=gr.Textbox(label='样本 ID')
                info=gr.Textbox(label='样本摘要', lines=8)
                prompt=gr.Textbox(label='训练输入结构', lines=10)
                chosen=gr.Textbox(label='采用回复 / 可改写', lines=3)
                rejected=gr.Textbox(label='对比拒绝回复', lines=3)
                load_status=gr.HTML()
                load.click(load_sample, inputs=idx, outputs=[sid,info,prompt,chosen,rejected,load_status])
                save=gr.Button('保存人工覆盖样本', variant='primary')
                save_out=gr.HTML()
                save.click(save_override, inputs=[sid,chosen,chosen,rejected], outputs=save_out)
            with gr.Tab('4. 训练中心'):
                gr.HTML("<div class='glass'><div class='section-title'>最终训练数据与 Baseline</div><div class='hint'>训练使用“自动原始样本 + 人工覆盖样本”的合并结果。人工覆盖不是必需。</div></div>")
                export_btn=gr.Button('只生成最终 SFT / DPO 文件')
                train_btn=gr.Button('一键生成最终数据并训练 Baseline', variant='primary')
                export_out=gr.HTML(); export_stat=gr.Textbox(lines=8, show_label=False)
                export_btn.click(export_final, outputs=[export_out, export_stat])
                train_out=gr.Textbox(lines=20, show_label=False)
                train_btn.click(train_baseline, outputs=train_out)
                gr.HTML("<div class='glass'><div class='section-title'>LoRA / SFT 微调配置</div><div class='hint'>这里生成本地或 AutoDL 可执行训练命令。真正 LoRA 训练建议在 Linux CUDA / AutoDL 环境执行。</div></div>")
                base=gr.Textbox(value='Qwen/Qwen2.5-1.5B-Instruct', label='基础模型 / 模型仓库 ID')
                model_source=gr.Dropdown(['ModelScope（推荐，适合 AutoDL 国内网络）','HuggingFace 官方/镜像','本地模型路径'], value='ModelScope（推荐，适合 AutoDL 国内网络）', label='模型下载来源')
                local_model_path=gr.Textbox(value='', label='本地模型路径（可选；如 /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct）')
                hf_endpoint=gr.Textbox(value='', label='HuggingFace 镜像地址（可选；例如 https://hf-mirror.com）')
                train_file=gr.Textbox(value='data/export/final_train_sft.jsonl', label='SFT 训练文件')
                out=gr.Textbox(value='outputs/lora_sft', label='输出目录')
                with gr.Row():
                    epochs=gr.Number(value=1, label='训练轮数')
                    lr=gr.Number(value=0.0002, label='学习率')
                    r=gr.Number(value=8, label='LoRA r')
                    alpha=gr.Number(value=16, label='LoRA alpha')
                cfg_btn=gr.Button('保存 LoRA 配置并生成训练命令')
                cfg_out=gr.HTML(); cmd_out=gr.Textbox(lines=8, label='训练命令')
                cfg_btn.click(save_train_config, inputs=[base,model_source,local_model_path,hf_endpoint,train_file,out,epochs,lr,r,alpha], outputs=[cfg_out,cmd_out])
            with gr.Tab('5. 云端训练'):
                gr.HTML("<div class='glass'><div class='section-title'>AutoDL 云端 LoRA 训练控制台</div><div class='hint'>本页面通过 SSH 控制你已经开机的 AutoDL 实例：上传数据、安装依赖、启动 LoRA 训练、读取日志、云端推理。密码只在本次会话中使用，不会写入文件。</div></div>")
                with gr.Row():
                    host = gr.Textbox(value='connect.bjb2.seetacloud.com', label='SSH Host')
                    ssh_port = gr.Number(value=24024, label='SSH Port')
                    username = gr.Textbox(value='root', label='用户名')
                password = gr.Textbox(value='', type='password', label='SSH 密码（不保存）')
                code_dir = gr.Textbox(value=DEFAULT_REMOTE_CODE_DIR, label='远程代码目录')
                asset_dir = gr.Textbox(value=DEFAULT_REMOTE_ASSET_DIR, label='远程资产与输出目录')
                gr.Markdown('**路径策略：** 代码固定放 `/root/SoulNPC-Agent/`；训练数据、配置、日志、LoRA 输出固定放 `/root/autodl-tmp/SoulNPC-Agent-assets/`。  \n**你的 AutoDL 端口映射：** 6006 -> https://u815652-9973-5b0ca6e7.bjb2.seetacloud.com:8443 ；6008 -> https://uu815652-9973-5b0ca6e7.bjb2.seetacloud.com:8443')
                with gr.Row():
                    env_btn = gr.Button('1. 检查云端环境', variant='secondary')
                    pack_btn = gr.Button('2. 生成本地训练包', variant='secondary')
                    upload_btn = gr.Button('3. 上传项目与数据', variant='primary')
                cloud_out = gr.Textbox(lines=18, label='云端操作反馈')
                pkg_path = gr.Textbox(value='', visible=False)
                env_btn.click(autodl_check_env, inputs=[host, ssh_port, username, password], outputs=cloud_out)
                pack_btn.click(make_remote_package, inputs=[asset_dir], outputs=[cloud_out, pkg_path])
                upload_btn.click(autodl_upload_project, inputs=[host, ssh_port, username, password, code_dir, asset_dir], outputs=cloud_out)

                gr.HTML("<div class='glass'><div class='section-title'>基础模型准备</div><div class='hint'>训练前必须保证基础模型已经在 AutoDL 本地可用。推荐使用 ModelScope 下载到 /root/autodl-tmp/models，然后训练时只从本地路径加载，避免 HuggingFace 网络失败。</div></div>")
                cloud_model_id = gr.Textbox(value='Qwen/Qwen2.5-1.5B-Instruct', label='基础模型 ID')
                cloud_model_source = gr.Dropdown(['ModelScope（推荐，适合 AutoDL 国内网络）','HuggingFace / 镜像','仅检查本地路径'], value='ModelScope（推荐，适合 AutoDL 国内网络）', label='模型下载来源')
                cloud_local_model_dir = gr.Textbox(value='/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct', label='云端本地模型目录')
                cloud_hf_endpoint = gr.Textbox(value='https://hf-mirror.com', label='HuggingFace 镜像地址（可选，仅 HuggingFace 模式使用）')
                with gr.Row():
                    check_model_btn = gr.Button('4. 检查基础模型')
                    prepare_model_btn = gr.Button('5. 下载 / 准备基础模型', variant='primary')
                check_model_btn.click(autodl_check_base_model, inputs=[host, ssh_port, username, password, asset_dir, cloud_local_model_dir], outputs=cloud_out)
                prepare_model_btn.click(autodl_prepare_base_model, inputs=[host, ssh_port, username, password, code_dir, asset_dir, cloud_model_id, cloud_model_source, cloud_local_model_dir, cloud_hf_endpoint], outputs=cloud_out)

                gr.HTML("<div class='glass'><div class='section-title'>训练控制</div><div class='hint'>首次运行建议先安装依赖，再准备基础模型，最后启动 SFT LoRA。系统会优先使用 tmux；如果云端没有 tmux，会自动切换到 nohup 后台运行，关闭本地网页不会中断云端训练。</div></div>")
                with gr.Row():
                    dep_btn = gr.Button('6. 安装云端训练依赖')
                    sft_btn = gr.Button('7. 启动 SFT LoRA 训练', variant='primary')
                    check_btn = gr.Button('检查训练状态')
                with gr.Row():
                    log_lines = gr.Number(value=120, label='读取日志行数')
                    log_btn = gr.Button('读取训练日志')
                dep_btn.click(autodl_install_deps, inputs=[host, ssh_port, username, password, code_dir], outputs=cloud_out)
                sft_btn.click(autodl_start_sft, inputs=[host, ssh_port, username, password, code_dir, asset_dir], outputs=cloud_out)
                check_btn.click(autodl_check_training, inputs=[host, ssh_port, username, password, asset_dir], outputs=cloud_out)
                log_btn.click(autodl_read_log, inputs=[host, ssh_port, username, password, asset_dir, log_lines], outputs=cloud_out)

                gr.HTML("<div class='glass'><div class='section-title'>云端推理</div><div class='hint'>训练完成后可以通过 SSH 命令直接推理，也可以启动 6006 端口的 Gradio 推理服务。</div></div>")
                infer_prompt = gr.Textbox(lines=8, label='推理输入 Prompt', value='角色：艾拉。事件：玩家违背了承诺但现在道歉。状态：信任下降、压力上升、情绪失望。行为意图：询问解释。请生成一句自然克制的中文 NPC 台词。')
                with gr.Row():
                    infer_btn = gr.Button('云端执行一次 LoRA 推理')
                    service_port = gr.Number(value=6006, label='服务端口')
                    serve_btn = gr.Button('启动云端推理服务')
                infer_btn.click(autodl_infer, inputs=[host, ssh_port, username, password, code_dir, asset_dir, infer_prompt], outputs=cloud_out)
                serve_btn.click(autodl_start_server, inputs=[host, ssh_port, username, password, code_dir, asset_dir, service_port], outputs=cloud_out)

            with gr.Tab('6. 角色推理体验'):
                gr.HTML("<div class='glass'><div class='section-title'>角色推理体验</div><div class='hint'>当前为规则版 Agent Loop；后续可接入微调后的 LoRA 模型替代台词生成层。</div></div>")
                event=gr.Dropdown(list(PRESET_COMPLEX_EVENTS.keys()), value='玩家违背了承诺', label='玩家事件')
                text=gr.Textbox(value='抱歉，我昨天说会回来，但我没有做到。', label='玩家台词')
                run=gr.Button('执行角色推理', variant='primary')
                dialogue=gr.Textbox(lines=4, label='NPC 回复')
                state=gr.Textbox(lines=16, label='角色状态')
                memory=gr.Textbox(lines=4, label='行为与记忆')
                run.click(infer, inputs=[event,text], outputs=[dialogue,state,memory])
    return demo

if __name__ == '__main__':
    build_app().launch(server_name='127.0.0.1', server_port=7860, show_error=True)
