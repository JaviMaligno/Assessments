from flask import Flask, render_template, request, jsonify
from data_utils import results
app = Flask(__name__)
#app.config["EXPLAIN_TEMPLATE_LOADING"] = True
@app.route('/')
def input_patient():
   return render_template('input_patient.html')

@app.route('/result', methods = ["POST"])
def return_results():
    if request.method == "POST":
        age, weight, height, sex = request.form.values()
        age, height, weight = float(age), float(height), float(weight)
    n_control, data_control, n_drug, data_drug = results(age,height,weight,sex)
    #create a template that contains 2 tables, one for control and another one for drug. 
    #each of them under a title and a sentence saying how many similar patients were found in each.
    #fill the table with percentages for each combination of response and ae
    #should I include 0? 
    #i could just have a dictionary with all the possibilities and use yy,nn, etc (defaultdict float)
    # Instead of a table maybe just items work just as fine
    # Ideally a pie chart
    return render_template("results.html", data_drug = data_drug, n_drug=n_drug, data_control=data_control, n_control=n_control)
   
if __name__ == "__main__":
    app.run(debug=True)
