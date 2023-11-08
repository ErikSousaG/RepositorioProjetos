import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize

#verifica se os recursos nltk estão atualizados
nltk.download("wordnet")
nltk.download("punkt")

#situacional:
os.environ['REQUESTS_CA_BUNDLE'] = 'paloalto.uniube.br.crt' #instancia o certificado uniube, para chamada de api externa

class treinamento: #classe de treinamento
    def __init__(self): #incia as variáveis
        self.dados = pd.read_csv('dados.csv') #lê o arquivo de treinamento 
        """ self.dados = self.dados[:int(len(self.dados)/2)] """  #lê metade dos arquivos, casa o treinamento fique muito pesado            
        self.X_treino = [self.preprocessa(texto) for texto in self.dados['texto']] #preprocessa os textos para treinamento
        self.model = self.constroi_modelo() #constroi o modelo de ml
        self.y_treino = self.dados['nota'] #salvas a notas dadas em uma variavel
        self.tamanho = 0 #variável que guarda o tamanho da redação
        self.tentativas = 0 #variável que guarda a qtde de tentativas de treinamento, utilizada para evitar que o treino ocorra infinitamente
        
    def preprocessa2(self,nota): #função que preprocessa as notas, caso haja número quebrado
        nt = str(nota)        
        nt = nt.replace(",",".")               
        return pd.to_numeric(nt)
    
    def preprocessa(self, texto): #funcao que transforma cada palava do texto em tokens, além de salvar o tamanho dela
        if isinstance(texto, str) and not None:
            tokens = word_tokenize(texto, language="portuguese")
            self.tamanho = len(tokens)
            return " ".join(tokens)
        else:
            return ""
    
    def constroi_modelo(self): #funcao de contrução do modelo
        
        tokenizador = tf.keras.layers.TextVectorization(max_tokens=1024, output_mode='int') #instancia o dicionário
        tokenizador.adapt(self.X_treino)
        
        def ativacao_zeroacem(x): #função de ativação personalizada da última camada da rede neural, para que retone uma nota de 0 a 100
            return tf.clip_by_value(x,0,100)
        
        tf.keras.utils.get_custom_objects()['ativacao_zeroacem'] = ativacao_zeroacem #faz um get na função de ativação personalizada

        #instancia o modelo e adiciona as camadas da rede em sequência
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(len(tokenizador.get_vocabulary()),2048)) #camada 1: neurônios de classificação
        model.add(tf.keras.layers.Conv1D(256,3,activation='relu')) #camada 2: neurônios de convolução unidimensional
        model.add(tf.keras.layers.GlobalAvgPool1D())  #camada 3: neurônio de conversão mediana para uma dimensão
        model.add(tf.keras.layers.Dense(2048, activation='elu')) #camada 4: neurônios com função de ativação linear exponêncial        
        model.add(tf.keras.layers.Dense(1, activation=ativacao_zeroacem)) #camada 5: neurônio único de saída com função de ativação personalizada, criada anteriormente

        #compila o modelo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), #define o otimizador keras e sua taxa de aprendizado
            loss=tf.keras.losses.mean_squared_error, #define a perda a ser medida, para nosso processo será necessário o mean squared error(erro médio quadrático)
            metrics=['mae','mse'] #define as métricas, MAE(Mean Absolute Error[Erro Médio Absoluto]) e MSE(Mean Squared Error[Erro Médio Quadrático])
        )
        return model #retorna o modelo na chamada da função 
    
    def treina_modelo(self, epochs=10000): #função de treinamento
        
        self.y_treino = pd.to_numeric(self.y_treino, errors='coerce') #transforma as notas em numéricas, caso não seja
                
        tokenizador = tf.keras.layers.TextVectorization(max_tokens=1024, output_mode='int') #instancia o dicionário
        tokenizador.adapt(self.X_treino)
        
        self.tentativas += 1 #aumenta o número de tentativas
        
        redacao_vetor = tokenizador(self.X_treino) #tokeniza a redacao com base no dicionário       
            
        #treinamento
        history = self.model.fit(redacao_vetor, #redação
                    self.y_treino, #nota
                    epochs=epochs, #número de épocas
                    batch_size=32, #número de pacotes a ser treinados por vez                                                          
                    validation_split=0.2, #separa uma parte dos dados para validação
                    callbacks=tf.keras.callbacks.EarlyStopping(monitor="mae",patience=6), #cria um callback que para o automaticamente o treinamento caso não haja evolução na métrica monitorada
                    verbose=1) #mostra cada época no console       

        #na variável history, facará salvo o histórico de treinamento em um dataframe com todos os parâmetros

        if len(history.history['loss']) <= 7 and self.tentativas <= 5: #verifica se há estagnação no processo de treinamento e tenta novamente

            print("erro de treinamento, tentando novamente")
            self.treina_modelo()

        elif self.tentativas > 5: #verifica se foi excedido o máx de tentativas
            
            print("Max tentantivas excedidas")

        else: #caso não haja erros, plota o gráfico das métricas de erro e salva o modelo em um arquivo "modelo.keras"
            def plot_history(history): #função de plotagem do gráfico        
                hist = pd.DataFrame()        
                hist["mae"] = history.history['mae']
                hist["val_mae"] = history.history['val_mae']
                hist["mse"] = history.history['mse']
                hist["val_mse"] = history.history['val_mse']

                plt.figure()
                plt.xlabel('Época')
                plt.ylabel('Erro Absoluto Médio [EAM]')
                plt.plot(hist['mae'],
                        label='Erro Treino')
                plt.plot(hist['val_mae'],
                        label = 'Erro Val')
                plt.ylim([0,30])
                plt.legend()

                plt.figure()
                plt.xlabel('Época')
                plt.ylabel('Erro Quadrático Médio[$EAM^2$]')
                plt.plot(hist['mse'],
                        label='Erro Treino')
                plt.plot(hist['val_mse'],
                        label = 'Erro Val')
                plt.ylim([0,1000])
                plt.legend()
                plt.show()

            plot_history(history) #plota o gráfico
            self.model.save("modelo.keras") #salva o modelo          

#controle de execução        
if __name__ == "__main__": #essa parte do código é executada apenas quando o arquivo Python é executado diretamente, e não quando o arquivo é importado como um módulo em outro programa ou script.         
    treinamento_inst = treinamento()
    treinamento_inst.treina_modelo()                


