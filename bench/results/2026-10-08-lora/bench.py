# bench: decode tok/s and prompt-read ms, LoRA on / off / no adapter; MTP or not.  Parses the engine's serve lines.
import subprocess, sys, os, re, random
from tokenizers import Tokenizer
W = "/workspace/heretic-work"; EXE = f"{W}/tools/strata-lora/build/strata"; G = f"{W}/gguf/strata"
tok = Tokenizer.from_file(f"{W}/runs/export-ara/model/tokenizer.json")
mtp = sys.argv[1] == "mtp"
def chat(u): return f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
random.seed(1)
words = open(f"{W}/tools/strata-lora/README.md").read().split()
def long_prompt(i):   # ~2K tokens, different per request (no cache reuse)
    start = random.randrange(0, len(words) - 1500)
    return chat(f"[{i}] Summarize this text in five bullet points:\n\n" + " ".join(words[start:start + 1400]))
gen = chat("Write a Python class implementing an LRU cache with get/put, then explain its complexity in detail.")
base = [EXE, "--serve", "--pack", f"{G}/pack", "--native", f"{G}/ara-trial9-IQ2_XS.gguf", "--ple-gguf",
        f"{G}/ara-trial9-IQ2_XS.gguf", "--max-context", "8192", "--spec", "4", "--prefill", "auto",
        "--expert-profile", f"{W}/tools/strata-lora/data/expert-profile.bin", "--expert-cache", "auto",
        "--prompt-cache", "0", "--gpu", os.environ.get("GPU", "3")]
if mtp: base += ["--mtp", f"{G}/mtp-bf16/rt"]
pat = re.compile(r"prompt (\d+) tokens = \d+ reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms")
def run(extra, keys, label):
    log = f"/tmp/bench-{label}-{mtp}.log"
    p = subprocess.Popen(base + extra, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "w"), text=True, bufsize=1)
    while not (l := p.stdout.readline()).startswith("READY"):
        if not l: sys.exit(f"engine died: {log}")
    n = 0
    for key in keys:
        for rep in range(4):   # rep 0: warm-up, not counted
            for ids, mx in ((tok.encode(long_prompt(n), add_special_tokens=False).ids, 1), (tok.encode(gen, add_special_tokens=False).ids, 256)):
                n += 1
                p.stdin.write(f"GEN {mx}{key} {','.join(map(str, ids))}\n")
                while not (l := p.stdout.readline()).startswith(("DONE", "ERR")): pass
    p.stdin.write("QUIT\n"); p.wait(timeout=120)
    rows = [tuple(map(int, m.groups())) for m in pat.finditer(open(log).read())]
    out, i = {}, 0
    for key in keys:
        r = rows[i:i + 8]; i += 8
        pre = [x for x in r[2::2]]; dec = [x for x in r[3::2]]   # skip the warm-up pair
        out[key or "none"] = (sum(x[2] for x in pre) / 3, sum(x[1] for x in pre) / 3,
                              sum(x[3] for x in dec) * 1000 / sum(x[4] for x in dec))
    return out
res = run(["--lora", f"{W}/gguf/ara-trial9-lora-F32.gguf"], [" lora=1", " lora=0"], "lora")
res.update(run([], [""], "none"))
print(f"{'mtp' if mtp else 'no mtp'}: setting | prompt read ms (tokens) | decode tok/s")
for k, (ms, t, ts) in res.items(): print(f"  {k:8} | {ms:7.1f} ({t:.0f}) | {ts:6.1f}")
