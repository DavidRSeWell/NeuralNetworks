

import dash
import dash_html_components as html
import base64

app = dash.Dash()

image_filename = '/Users/befeltingu/NeuralNetworks/DB/images/rbm/predicted_images.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div([

    html.Img(src='data:image/png;base64,{}'.format(encoded_image))
])

if __name__ == '__main__':

    app.run_server(debug=True)