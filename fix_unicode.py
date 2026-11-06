import os
import glob

replacements = {
    "\u2500": "-", "\u2550": "=", "\u2013": "-", "\u2014": "--",
    "\u2192": "->", "\u2190": "<-", "\u2714": "[OK]", "\u2718": "[X]",
    "\u2713": "[OK]", "\u2717": "[FAIL]", "\u25ba": ">",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2015": "-", "\u00e2\u0080\u0093": "-",
}

files = glob.glob(r"d:\Healthcare_AgenticSystem\src\**\*.py", recursive=True)
files += glob.glob(r"d:\Healthcare_AgenticSystem\evaluation\*.py")
files += [r"d:\Healthcare_AgenticSystem\run_demos.py"]

fixed = 0
for fp in files:
    try:
        with open(fp, "rb") as f:
            content = f.read().decode("utf-8")
        new_content = content
        for old, new in replacements.items():
            new_content = new_content.replace(old, new)
        if new_content != content:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"  Fixed: {os.path.basename(fp)}")
            fixed += 1
    except Exception as e:
        print(f"  Skip {fp}: {e}")

print(f"\nTotal files fixed: {fixed}")
