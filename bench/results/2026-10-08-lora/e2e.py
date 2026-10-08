# e2e: one engine with --lora, requests lora=1 / lora=0 / lora=1; one engine without; greedy tokens compared.
import subprocess, sys, json, time, os
from tokenizers import Tokenizer
W = "/workspace/heretic-work"
EXE = f"{W}/tools/strata-lora/build/strata"
G = f"{W}/gguf/strata"
tok = Tokenizer.from_file(f"{W}/runs/export-ara/model/tokenizer.json")
mtp = sys.argv[1] == "mtp"
def chat(u): return f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prompts = [chat("What is the capital of France? Answer in one sentence."),
           chat("Write a short Python function that checks whether a number is prime, then explain it. " * 6)]
ids = [tok.encode(p, add_special_tokens=False).ids for p in prompts]
base = [EXE, "--serve", "--pack", f"{G}/pack", "--native", f"{G}/ara-trial9-IQ2_XS.gguf", "--ple-gguf",
        f"{G}/ara-trial9-IQ2_XS.gguf", "--max-context", "4096", "--spec", "4", "--prefill", "auto", "--expert-profile", f"{W}/tools/strata-lora/data/expert-profile.bin", "--expert-cache", "auto", "--gpu", os.environ.get("GPU", "3")]
if mtp: base += ["--mtp", f"{G}/mtp-bf16/rt"]
def run(extra, reqs):
    p = subprocess.Popen(base + extra, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(f"/tmp/e2e-{len(extra)}-{mtp}.log", "w"), text=True, bufsize=1)
    while True:
        l = p.stdout.readline()
        if not l: sys.exit("engine died: see /tmp/e2e-*.log")
        if l.startswith("READY"): break
    out = []
    for key, i in reqs:
        p.stdin.write(f"GEN 48{key} {','.join(map(str, ids[i]))}\n")
        toks, info = [], ""
        while True:
            l = p.stdout.readline()
            if l.startswith("T "): toks.append(int(l[2:]))
            elif l.startswith("INFO"): info = l.strip()
            elif l.startswith("ERR"): print("ERR", l.strip()); break
            elif l.startswith("DONE"): break
        out.append(toks)
        print(f"{key or '(none)':>8} p{i}: {tok.decode(toks)[:110]!r}  [{info[:90]}]", flush=True)
    p.stdin.write("QUIT\n"); p.wait(timeout=120)
    return out
reqs = [(k, i) for i in (0, 1) for k in (" lora=1", " lora=0", " lora=1")]
a = run(["--lora", f"{W}/gguf/ara-trial9-lora-F32.gguf"], reqs)
b = run([], [("", 0), ("", 1)])
for i in (0, 1):
    on1, off, on2 = a[3*i:3*i+3]
    print(f"p{i}: on==on again {on1 == on2} | off==no-adapter {off == b[i]} | on!=off {on1 != off}")
