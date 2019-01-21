
import dash

import dash_html_components as html
import dash_core_components as dcc

import base64

from dash.dependencies import Input, Output

app = dash.Dash()

image_filename = '/Users/befeltingu/NeuralNetworks/DB/images/rbm/predicted_images.png' # replace with your own image

encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div([

    dcc.Input(id='my-id', value='initial value', type='file'),

    #html.Img(src='data:image/png;base64,{}'.format(encoded_image))
    html.Img(id='my-img',src='', width="2000",height="2000"),

    html.Div(id='file-name')

])


@app.callback(
    Output(component_id='my-img', component_property='src'),
    [Input(component_id='my-id', component_property='value')]
)
def upload_image(input_value):

    print(input_value)

    image_filename = '/Users/befeltingu/NeuralNetworks/CNN/data/' + str(input_value.split('\\')[-1])  # replace with your own image

    encoded_image = base64.b64encode(open(image_filename, 'rb').read())

    src = 'data:image/png;base64,{}'.format(encoded_image)

    #return "Hey this is the file path yo {file}".format(file=input_value)
    return src




if __name__ == '__main__':

    app.run_server(debug=True)