# %%
# t-SNE 
## Preparação dos dados
votacao_2021 = get_dataframe_votacao_senadores() # função criada para coletar dados de votações da API do senado
votacao = votacao_2021[['sigladescricaovoto', 'sessaoplenaria_datasessao', 'nome_completo']].set_index('sessaoplenaria_datasessao').sort_index()
votacao.sigladescricaovoto.replace(['Sim', 'Não'], [1, -1], inplace = True)
# %%
votacao_senadores = votacao.pivot_table(values='sigladescricaovoto', index=votacao.index, columns='nome_completo', aggfunc='first')
senadores = list(votacao_senadores)
# %%
import numpy as np
votacao_array = votacao_senadores.values
votacao_array_trans = np.transpose(votacao_array)
#%%
# modelo
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
tsne_features = model.fit_transform(votacao_array_trans)

#%%
# Gráfico
import matplotlib.pyplot as plt
xs = tsne_features[:,0]
ys = tsne_features[:,1]

fig = plt.figure(figsize=(22,22))
fig.patch.set_facecolor("#F4F4F4")

ax = fig.add_subplot(1,1,1)

ax.set_facecolor('#F4F4F4')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.scatter(xs, ys, alpha=0.4, color="#40E0D0")

for x, y, company in zip(xs, ys, senadores):
    plt.annotate(company, (x, y))

plt.show()