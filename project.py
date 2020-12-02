import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output,State
import base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

import sklearn.linear_model as skl_lm

###########
# use path of pictures 
path=" "

data = pd.read_csv("data/insurance3r2.csv")
############
data_corr= data.drop(["insuranceclaim","region","sex","smoker"], axis = 1)
corr_matrix_list = data_corr.corr().values.tolist()
x_axis = data_corr.corr().columns
y_axis = data_corr.corr().index.values


trace1 = go.Heatmap(x=x_axis ,y=y_axis, z=corr_matrix_list, colorscale='RdBu' )
data1 = [trace1]
layout1  = dict(title = "Correlation Matrix")
figure1 = dict(data=data1,layout=layout1)
#### Region
values=data.region.value_counts()
values=values.tolist()
labels2=["Northeast", "Northwest", "Southeast", "Southwest"]
trace2 = go.Pie(labels=labels2, values=values)
layout2  = dict(title = "Distribution of region")

data2 = [trace2]
figure2 = dict(data=data2,layout=layout2)


###############


test_png2 = 'pic4.png'
test_png3 = 'pic5.png'
test_png4 = 'pic6.png'

test_base64_2 = base64.b64encode(open(test_png2, 'rb').read()).decode('ascii')
test_base64_3 = base64.b64encode(open(test_png3, 'rb').read()).decode('ascii')
test_base64_4 = base64.b64encode(open(test_png4, 'rb').read()).decode('ascii')




###############


list_of_images = [
    'Sex-Steps.png',
    'Smoker-Sex.png',
    'Smoker-Steps.png',
    'Age-Sex.png',
    'Sex-Bmi.png'
]



list_of_images2=[
"Confusion matrix.png",
"Linear regression of bmi and charges.png",
"Linear regression of steps and charges.png"

]


#################


app = dash.Dash()
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})


app.layout = html.Div([
	html.Div([
    html.H1('Health insurance')]),
    html.Div([
    html.H4('Introduction')]),
    html.Div([
		html.P(''' This is "Sample Insurance Claim Prediction Dataset". The goal is to learn features, as much information as possible and predict 
			probability of a claim. At first, let's understand what we have: '''),
			html.P('''- age : age of policyholder sex: gender of policy holder (female=0, male=1) '''),
			html.P('''- bmi : Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, 
			objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 25 , '''),
			html.P('''- steps : average walking steps per day of policyholder , '''),
			html.P('''- children : number of children / dependents of policyholder , '''),
			html.P('''- smoker : smoking state of policyholder (non-smoke=0;smoker=1) , '''),
			html.P('''- region : the residential area of policyholder in the US (northeast=0, northwest=1, southeast=2, southwest=3) , '''),
			html.P('''- charges : individual medical costs billed by health insurance, '''),
			html.P('''- insuranceclaim : yes=1, no=0. 


								 ''')]),
	html.Div([
		html.P('''  Now let's understand are they linearly dependent on features by plotting correlation matrix(we use Pearson correlation). 



		 ''')]),
   
	html.Div([dcc.Graph(id='fig1', figure = figure1)]



					, className = "row"),
	html.Div([
		html.P(''' From the correlation matrix, we can see that "Bmi" and "Average walking steps " have the highest negative dependence.


								 '''),
		 html.Div([
    html.H4('Features description')]),
		html.P(''' As I don't sure that "Region" is an important feature, let's learn its distribution.


								 ''')]),

	html.Div([html.Div(dcc.Graph(id='fig2', figure = figure2),className = "six columns"),
		html.Div( html.Img(src='data:image/png;base64,{}'.format(test_base64_2)),className = "six  columns"),


		]

					, className = "row"),

	html.Div([
		html.Div( html.Img(src='data:image/png;base64,{}'.format(test_base64_3)),className = "six  columns"),
html.Div( html.Img(src='data:image/png;base64,{}'.format(test_base64_4)),className = "six columns"),

		]

					, className = "row"),
	html.Div([
		html.P('''From the pie chart, we can say that we have partly the same amount of observations for all regions.
			From 'Distribution of average walking steps in the region' plot we want to understand if the region affects on steps distribution. Only  for  "Southeast"  
			distribution is different.
			With 'Distribution of gender in region ' plot we want to check if the proportion of males and females is different from region to region.
			In the last plot, we check the same but for smokers and non-smokers.

		 '''),
		html.P(''' As a conclusion, there is no so much difference between regions but it can be a useful feature in regression.


		     ''')
		]),

	html.Div([
		html.P(''' Now, let's learn some combinations via plots.


								 ''')]),
	html.Div([
    dcc.Dropdown(
        id='image-dropdown',
        options=[{'label': i.split('.')[0], 'value': i} for i in list_of_images],
      
        value=list_of_images[0]
    ),
    html.Img(id='image')
]),
	html.Div([
		html.P(''' From "Smoker-Sex" plot we can say that 17.5% of women are smokers 22.5% of men are smokers.We also can say that smokers walk
		 less than non-smokers ,it's obvious and so on .


								 '''),
		html.Div([
    html.H4('Prediction')]),
		html.P(''' The goal was to predict claim, we will use logistic regression for it .Data were split to test and train, 
			then I fit the model on train data and predict	for test data for checking efficiency of model. Applying on test data we get 88% accuracy.
			Below we can see confusion matrix , and some linear regressions 
		for vizualization dependance of variables.


								 ''')]),

html.Div([
    dcc.RadioItems(
        id='image-RadioItems',
        options=[{'label': i.split('.')[0], 'value': i} for i in list_of_images2],
        
        value=list_of_images2[0],labelStyle={'display': 'inline-block'}
    ),
    html.Img(id='image1')
]),
html.Div([
    html.H4('Conclusion')]),
html.Div([
		html.P(''' Put your name in the bottom below, please.


								 ''')]),
html.Div([
    dcc.Input(id='text_in', value='Name', type='text'),
    html.Button(id='submit',n_clicks=0, children='Submit'),
    html.Div(id='text_out')])

			],


	className = "container")
 


@app.callback(
    Output('image', 'src'),
    [Input('image-dropdown', 'value')
    ]
)

def update_image_src(image_path):
    
    print('current image_path = {}'.format(image_path))
    encoded_image = base64.b64encode(open(image_path, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(
    Output('image1', 'src'),
    [Input('image-RadioItems', 'value')
    ]
)


def update_image_src1(image_path):
   
    print('current image_path = {}'.format(image_path))
    encoded_image = base64.b64encode(open(image_path, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())

@app.callback(
	Output(component_id='text_out', component_property='children'),
    [Input(component_id='submit', component_property='n_clicks')],
    [State(component_id='text_in', component_property='value')]
)
def  update_text_output(clicks,input_value_1):
    return "Congratulations "+input_value_1 + "!!! You learn much about health insurance and now by having some features you can predict future claim. " 





if __name__ == '__main__':
    app.run_server(debug=True)


