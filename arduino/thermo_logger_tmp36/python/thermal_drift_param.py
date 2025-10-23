# thermal_drift_param.py
import numpy as np, matplotlib.pyplot as plt
alpha={"Al6061":23e-6,"MgAZ31":26e-6,"CFRP":2e-6}
L0=30e-3; dT=np.linspace(0,80,81)
for name,a in alpha.items():
    plt.plot(dT,(L0*a*dT)*1e6,label=name)
plt.xlabel("ΔT (°C)"); plt.ylabel("Focus shift (µm)")
plt.title("Material choice vs thermal focus drift"); plt.legend(); plt.show()
