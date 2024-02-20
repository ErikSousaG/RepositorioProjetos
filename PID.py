import tensorflow as tf
import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

Npop = 50
Ngenes = 3
Fmut = 0.8
Nepocas = 100

numh = [14]
denh = [1,2,4]
Gs = ctrl.TransferFunction(numh,denh)

def FTpid(Kp,Ki,Kd):
    numpid = [float(Kd), float(Kp), float(Ki)]
    denpid = [1,0]
    return ctrl.TransferFunction(numpid,denpid)

def FTsis(FTpid):
    sis = ctrl.series(Gs,FTpid)
    print(sis)
    return sis

def fitness(melhor_param):
    
    Kp = melhor_param[0]
    Ki = melhor_param[1]
    Kd = melhor_param[2]
    pid = FTpid(Kp,Ki,Kd)  
    time = np.arange(0, 15, 0.1)    
    resposta = ctrl.step_response(ctrl.feedback(FTsis(pid),1),T=time) 
    overshoot = abs(np.max(resposta.outputs) - 1)
    return overshoot
    
pop = tf.random.uniform(shape=(Npop,Ngenes),minval= 0.1,maxval=5)
melhor_param_todos = tf.random.uniform(shape=(1,Ngenes),minval= 0.1,maxval=5).numpy()
melhor_param_todos = melhor_param_todos[0]

plt.ion()
fig, ax = plt.subplots()
linhas = ax.plot([],[])
ax.set_xlabel('Tempo')
ax.set_ylabel('Resposta do Sistema')
ax.set_title('Resposta do Sistema')
plt.grid(True)

for epoca in range(Nepocas):
    
    VFit = [fitness(individuo) for individuo in pop]
    
    SelIndice = tf.argsort(VFit, direction='DESCENDING')
    SelPop = tf.gather(pop, SelIndice[:Npop])
    
    CrossIndice = tf.random.shuffle(tf.range(Npop))
    CrossPop = tf.gather(SelPop,CrossIndice)
    
    MMasc = tf.random.uniform(shape=(Npop,Ngenes)) < Fmut
    MPop = tf.random.uniform(shape=(Npop, Ngenes), minval=0.1, maxval=5)
    PopM = tf.where(MMasc,MPop,CrossPop)
    
    pop = tf.concat([SelPop[int(Npop/2):], PopM[:int(Npop/2)]], axis=0)

    melhor_param_index = tf.argmin([fitness(param) for param in pop])
    melhor_param = pop[melhor_param_index].numpy()
    
    if fitness(melhor_param) < fitness(melhor_param_todos):
        melhor_param_todos = melhor_param

    melhorKp = melhor_param_todos[0]
    melhorKi = melhor_param_todos[1]
    melhorKd = melhor_param_todos[2]    
    melhorpid = FTpid(melhorKp,melhorKi,melhorKd)
        
    sistema = FTsis(melhorpid)
    time = np.arange(0, 15, 0.1)
    tempo, resposta = ctrl.step_response(ctrl.feedback(sistema,1),T=time)
        
    linhas[0].set_xdata(tempo)
    linhas[0].set_ydata(resposta)
    ax.relim()
    ax.autoscale_view()
    ax.set_title(f'Resposta ao Degrau - Iteração {epoca}')
    fig.canvas.draw()
    fig.canvas.flush_events()

plt.ioff()
plt.show()

print("Melhores Parâmetros do PID:", melhor_param_todos)
print("Overshoot Mínimo:", fitness(melhor_param_todos))