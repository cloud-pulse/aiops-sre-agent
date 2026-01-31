## 1
`````

python -m venv .venv

source .venv/Scripts/activate

python -m pip install --upgrade pip setuptools wheel

$ python - <<EOF
import tomllib
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]
open("requirements.txt","w").write("\n".join(deps))
print("requirements.txt created")
EOF

 pip install -r requirements.txt
````