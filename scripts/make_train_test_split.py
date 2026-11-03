"""
Split the slice folders into training_data/ and testing_data/ (by block, not at random).

Why by block: neighbouring CT slices are almost identical, so a random split would
leak information from train to test. Slices 0-159 are used to CHOOSE tool settings
("training"); slices 160-319 are only used to SCORE them ("testing").

Usage:
    python scripts/make_train_test_split.py data/set1 submission_data
    python scripts/make_train_test_split.py data/set1 submission_data --link   # hard-links, no extra disk
Result:
    submission_data/training_data/{Assembly,Ceramic,Metal,Braze,Outer}/...
    submission_data/testing_data/{Assembly,Ceramic,Metal,Braze,Outer}/...
"""
import os, re, sys, glob, shutil

SPLIT_AT = 160
NAMES = ["Assembly", "Ceramic", "Metal", "Braze", "Outer"]


def key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p))]


def main(src, dst, link=False):
    for name in NAMES:
        files = sorted(glob.glob(os.path.join(src, name, "*.tif")), key=key)
        if not files:
            print(f"!! no files in {os.path.join(src, name)}"); continue
        for i, f in enumerate(files):
            part = "training_data" if i < SPLIT_AT else "testing_data"
            out_dir = os.path.join(dst, part, name)
            os.makedirs(out_dir, exist_ok=True)
            target = os.path.join(out_dir, os.path.basename(f))
            if os.path.exists(target):
                continue
            if link:
                os.link(f, target)
            else:
                shutil.copy2(f, target)
        print(f"{name}: {min(SPLIT_AT, len(files))} training, {max(len(files) - SPLIT_AT, 0)} testing")
    print("Done ->", dst)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], link="--link" in sys.argv)
