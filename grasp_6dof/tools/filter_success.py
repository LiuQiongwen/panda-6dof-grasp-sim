import json

in_path = "grasp_6dof/out/cylinder_0p08_grasps_validated.json"
out_path = "grasp_6dof/out/cylinder_0p08_grasps_success.json"

with open(in_path, "r") as f:
    data = json.load(f)

succ = [g for g in data if g.get("success")]
print(f"Total={len(data)}, success={len(succ)}")

with open(out_path, "w") as f:
    json.dump(succ, f, indent=2)

print("Saved:", out_path)

