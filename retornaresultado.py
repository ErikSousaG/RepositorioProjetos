import tensor_treino
from tensor_treino import tf 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#para usar tratar dados em excel será necessário a biblioteca openpyxl(pip install openpyxl)

treinamento = tensor_treino.treinamento() #instancia a classe treinamento

dados2 = pd.read_csv("valida.csv") #lê a redação
nota_real = dados2['nota'] #salva a nota atribuída a ela 

tokenizador = tf.keras.layers.TextVectorization(max_tokens=1024, output_mode='int') #instancia o dicionário de treinamento 
tokenizador.adapt(treinamento.X_treino) 
    
redacao = [treinamento.preprocessa(texto) for texto in dados2['texto']] #preprocessa a redação

redacao = tokenizador(redacao) #tokeniza a redação, de acordo com o dicionário

model = tf.keras.models.load_model("modelo.keras") #carrega o modelo de treinamento

nota_predita = np.array(model.predict(redacao)) #faz a previsão da nota
nota_predita = nota_predita.flatten() #tranforma a matriz numpy em um array unidimensional

data = {'Nota_Real': nota_real, 'Nota_Predita': nota_predita} #transforma a nota predita e a nota em um dataframe
df = pd.DataFrame(data)
 
df.to_excel('notas_preditas.xlsx', index=False) #joga a nota dada e a nota predita em uma tabela excel, para fins de comparação

#plota em um gráfico as notas dadas e preditas, para fins de comparação
plt.scatter(nota_real, nota_predita,c=np.random.rand(len(nota_predita),3))
plt.xlabel('Notas')
plt.ylabel('Predições')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()
