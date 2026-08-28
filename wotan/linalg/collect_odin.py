
import os

def collect_odin_files(root_dir, output_file):
    with open(output_file, "w", encoding="utf-8") as out:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".odin"):
                    full_path = os.path.join(dirpath, filename)
                    out.write(f"// ===== {full_path} =====\n")
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            out.write(f.read())
                            out.write("\n\n")
                    except Exception as e:
                        out.write(f"// ERROR reading {full_path}: {e}\n\n")

if __name__ == "__main__":
    project_root = "./"          # change if needed
    output_path = "all_odin.txt" # output file
    collect_odin_files(project_root, output_path)
    print(f"Collected Odin sources into {output_path}")
