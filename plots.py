import wandb
import matplotlib.pyplot as plt
from adjustText import adjust_text


api = wandb.Api()

runs = api.runs("Proj-IF702/miniprojeto2-bitcoin-lstm")

losses = []
pocids = []
run_names = []

for run in runs:
    has_target_tag = "03/10" in run.tags
    
    if (
        has_target_tag 
        and run.state == "finished" 
        and "best_val_mse" in run.summary 
        and "best_pocid_percent" in run.summary
    ):
        losses.append(run.summary["best_val_mse"])
        pocids.append(run.summary["best_pocid_percent"])
        run_names.append(run.name)

print(f"Total de runs carregadas com a tag '03/10': {len(losses)}")

# --- Cálculo da Fronteira de Pareto ---
pareto_losses = []
pareto_pocids = []
pareto_names = []

for i in range(len(losses)):
    is_pareto = True
    for j in range(len(losses)):
        if i != j:
            if (losses[j] <= losses[i] and pocids[j] >= pocids[i]) and (losses[j] < losses[i] or pocids[j] > pocids[i]):
                is_pareto = False
                break
    if is_pareto:
        pareto_losses.append(losses[i])
        pareto_pocids.append(pocids[i])
        pareto_names.append(run_names[i])

# --- Plot do Gráfico ---
plt.figure(figsize=(11, 7))

# Plota todos os trials
plt.scatter(losses, pocids, color='tab:gray', alpha=0.5, label='Outros Trials')

# Plota a Fronteira de Pareto
plt.scatter(pareto_losses, pareto_pocids, color='tab:red', s=90, edgecolors='black', label='Fronteira de Pareto', zorder=5)

# Linha conectando os pontos da fronteira
sorted_pareto = sorted(zip(pareto_losses, pareto_pocids, pareto_names), key=lambda x: x[0])
if sorted_pareto:
    px, py, pnames = zip(*sorted_pareto)
    plt.plot(px, py, 'r--', alpha=0.6, linewidth=1.5)

# --- ADICIONA TEXTOS E REAJUSTA AUTOMATICAMENTE ---
texts = []
for x, y, name in zip(pareto_losses, pareto_pocids, pareto_names):
    label = name.split('_')[1] if 'trial' in name else name
    texts.append(
        plt.text(
            x, y, label, 
            fontsize=9, 
            fontweight='bold', 
            color='darkred'
        )
    )


# Reposiciona automaticamente os textos para não se colidirem e desenha setas até os pontos
adjust_text(
    texts, 
    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
    expand_text=(1.2, 1.4)
)

plt.xlabel('Validation MSE (Minimizar) ↓', fontsize=11)
plt.ylabel('Validation POCID % (Maximizar) ↑', fontsize=11)
plt.title('Fronteira de Pareto', fontsize=13, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig("pareto_frontier_03_10.png", dpi=300)
plt.show()



