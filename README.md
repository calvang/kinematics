# Kinematics Python and TypeScript Modules

Python3 and TypScript implementations of kinematics for a custom 6-axis robotic arm. The 2D kinematic functions are applicable to any 2D constrained systems.

## Usage

To install dependencies for the Python3 module, type:

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

For the TypeScript module, you must have TypeScript installed, and preferably `ts-node`:

```bash
npm i -g typscript ts-node
```

To test that everything is working, run:

```bash
python3 test.py & ts-node test.ts
```
