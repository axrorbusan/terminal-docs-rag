"""Download the 19 SMDG guides and the embedding model, and check their SHA-256.

  python scripts/fetch_data.py

The PDFs are published free of charge by SMDG (https://smdg.org/documents/). They are not
stored in this repository; this script fetches the exact versions used for the results.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMDG = "https://smdg.org/wp-content/uploads/"

PDFS = {
    "MIGs/BAPLIE/BAPLIE3.1.1-02.pdf": "88c76b659352e09080e3c3e371bd752fd785d97af5f6522f2e842f9fe8081fb1",
    "MIGs/BAPLIE/Why-BAPLIE-3-2024-0.4.pdf": "6104cdc6ba9f405ab5aa6d626f132c9102acfd529e402cf728ba56bdda6a6df3",
    "MIGs/MOVINS/SMDG_MIG_MOVINS_D95B_2.1.2.pdf": "e050b1dd0a51fb5bc9e96e306650c8dec47e984ccf613b992273da146a3dadc5",
    "MIGs/ContainerMsgs/SMDG-COPRAR213.pdf": "52ae1718a684752bc4da99165dab2289c078be8ccaf85c738be65780f5662daa",
    "MIGs/ContainerMsgs/SMDG-COARRI213.pdf": "3f73d11c767d5bd9f9cb9bbd0278332a2fbd80174e96f3d0839ff45f042937d6",
    "MIGs/ContainerMsgs/SMDG-COPARN213.pdf": "0e3bde2d8d5a61bbb9963b8cb80cd5ad1bef581bcebbcf64511e3b95b1b7c495",
    "MIGs/ContainerMsgs/SMDG-CODECO213.pdf": "aebd3382e370356671e0e7c5194adf4f5c7b327cbe266e0bca879695183ae707",
    "MIGs/ContainerMsgs/COREOR20.pdf": "953e1d8370e7c9a8f82b37fe659b55fb75bb54aeaa1b2e74d6b7eda464a0d5ce",
    "MIGs/ContainerMsgs/COPINO20.pdf": "2fa10ef0567e8a6ec57fe493b925c9bc64c18c563e81ebbd811e85ee48f2ad73",
    "MIGs/VERMAS/SMDG_MIG_VERMAS_D24A_1.1.0.pdf": "d1a6bac7d66d9d4b95f9d1146fee35ad2cf4ac3c0d1e9474d0aca812d54173c2",
    "MIGs/Other/IFTSTA-SMDG.pdf": "cc535d3814ac43449c1c274b2545a76b291a1940002a3df3840edbfee61e4ce5",
    "MIGs/TPFREP/TPFREP41_20210525.pdf": "6dfe396711ef74688af477a88d9dffdf8fe9a7633e3b092decff185c28b51936",
    "MIGs/Other/DESTIM-D96B-Equipment-Damage-and-Repair-Estimate-Message-Implementation-Guide-SMDG.pdf": "95c7dc107648b97e5b4721124800bab181924e12f8ee668245f9d11c796eca1b",
    "Recommendations/SMDG-Recommendation-01.pdf": "91f3b33121a804e74ffd0647b165c4909f723348bf8576aa75cfd6f783adb955",
    "Recommendations/SMDG-Recommendation-02.pdf": "d79a29e4ef04f2edf251d75248b4fdfa0ef8721d3b937bbeb1755580f80d3d72",
    "Recommendations/SMDG-Recommendation-03.pdf": "7e4713f06bd8400efb7bcf91cebf4b30ccf684d1257a3781266c2254ba9acc79",
    "Recommendations/Recommendation-06-SMDG-Recommended-Messages-v1-20200220.pdf": "c4643c31dd2f4b98f97defc98ae988649ed17d695cccdefa890b97d012f13e74",
    "Recommendations/Recommendation-07-UsageOfCodeLists-v8-20231112.pdf": "628c6911b0d82486d6c1990565030ceb09144e3d9092e2583217de4bff1a4712",
    "Recommendations/SMDG-Recommendation-08-MT-Tank-with-Residue_v8.pdf": "b312346f42a8ed40d0862d6d402df4be8009184cf88bb22ba743ec30f721b7ab",
}

# all-MiniLM-L6-v2 as ONNX (Xenova export), pinned to one revision.
HF = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/"
MODEL = {
    "onnx/model.onnx": ("model.onnx", "759c3cd2b7fe7e93933ad23c4c9181b7396442a2ed746ec7c1d46192c469c46e"),
    "tokenizer.json": ("tokenizer.json", "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"),
}


def fetch(url, dest, sha):
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha:
        print(f"ok      {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "terminal-docs-rag/1.0"})
    data = urllib.request.urlopen(req, timeout=120).read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        print(f"CHANGED {dest.name}: sha256 {got[:12]}… differs from the version used for the results")
    dest.write_bytes(data)
    print(f"fetched {dest.name} ({len(data) / 1e6:.2f} MB)")
    return got == sha


def main():
    ok = True
    for path, sha in PDFS.items():
        ok &= fetch(SMDG + path, ROOT / "data" / "raw" / Path(path).name, sha)
    for path, (name, sha) in MODEL.items():
        ok &= fetch(HF + path, ROOT / "models" / "all-MiniLM-L6-v2" / name, sha)
    if not ok:
        print("Some files differ from the pinned versions; results may differ slightly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
