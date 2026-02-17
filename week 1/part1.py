import matplotlib.pyplot as plt
from package import load_rirs

print("Code is gestart... even geduld.") # Check of de code überhaupt draait

# 1. Laad je bestand (zorg dat de bestandsnaam klopt!)
path_to_file = 'rirs/rirs_20260211_142151.pkl' 
acousticScenario = load_rirs(path=path_to_file) 

print(f"Scenario geladen: {acousticScenario.roomDim}")

# 2. Haal de RIR op van bron 2 (index 1) naar microfoon 3 (index 2)
# Python telt vanaf 0: Bron 1=0, Bron 2=1 | Mic 1=0, Mic 2=1, Mic 3=2
rir = acousticScenario.RIRs_audio[1][2]

# 3. Plot de grafiek
plt.figure(figsize=(10, 4))
plt.plot(rir)
plt.title('RIR van Bron 2 naar Microfoon 3')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid(True)

# 4. SLA HET PLAATJE OP in plaats van tonen
output_file = 'mijn_rir_grafiek.png'
plt.savefig(output_file)
print(f"Klaar! Open nu het bestand '{output_file}' in je linker zijbalk.")

test = 1