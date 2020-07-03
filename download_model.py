from requests import get

if __name__ == '__main__':
    with open('./stored_models/model_weights.p', 'wb') as f:
        response = get('https://surfdrive.surf.nl/files/index.php/s/EuUqRp3VYLBmoBk/download')
        f.write(response.content)
        print('Done downloading weights.')