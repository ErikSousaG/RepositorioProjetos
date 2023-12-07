from flask import Flask, render_template, request
import retornaresultado


app = Flask(__name__,static_folder='static')

#criar pagina
#route
#funcao

@app.route("/", methods= ['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/mostraresultado", methods = ['POST'])
def processar_redacao():
    redacao = request.form['redacao']
    resultado = retornaresultado.retorna(redacao)                
    return render_template("mostraresultado.html",
                            resultado = resultado,
                            redacao = redacao)

#rodar o site
if __name__ == "__main__":
    app.run(debug=True)