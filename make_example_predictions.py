import requests

from toloka_monitoring.compute_monitoring_metrics import compute_monitoring_metrics

cat_urls = [
    'https://i.ytimg.com/vi/jH7e1fDcZnY/maxresdefault.jpg',
    'https://icatcare.org/app/uploads/2018/07/Thinking-of-getting-a-cat.png',
    'https://cdn.pixabay.com/photo/2014/11/30/14/11/cat-551554__340.jpg',
    'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTrQr9K0QO5CGjSj7pMAaA7ftCRoR4GB1Mlg&usqp=CAU',
    'https://lh3.googleusercontent.com/DQj-gonAVTlhj5W7_DhBVmX-0P42rfvx8TSp1WfQeZ6iFIon6InIS8M4Nbqy7Ql5ahgEXSiRDiWD88v-bcPYIEAg3Q=w640-h400-e365-rj-sc0x00ffffff',
]

dog_urls = [
    'https://post.medicalnewstoday.com/wp-content/uploads/sites/3/2020/02/322868_1100-800x825.jpg',
    'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSm--OTFbGraHEpOHWr0eW2gDn25IBhRfgqKeflwLtwO0n6ft09PdG8_W0V9HPaSMSAZOE&usqp=CAU',
    'https://akm-img-a-in.tosshub.com/indiatoday/images/story/202108/international_dog_day_2021_4_r_1200x768.jpeg?mhENil.rEsB2Wju30UDroUYKmJ4NfkX4&size=1200:675',
    'https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/chihuahua-dog-running-across-grass-royalty-free-image-1580743445.jpg',
    'https://media.istockphoto.com/photos/annoyed-boxer-picture-id140397195?k=20&m=140397195&s=612x612&w=0&h=5HS0lfB3zxeumDBUYAvL4A-Hhc9u5ntuFaR3xUxx-bo=',
]

other_urls = [
    'https://www.successful-city.com/wp-content/uploads/2021/09/22_main-v1632750930.jpg',
    'https://st.depositphotos.com/1011833/3557/i/600/depositphotos_35577647-stock-photo-wooden-arrow-sign-post-or.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/AH-64D_Apache_Longbow.jpg/1200px-AH-64D_Apache_Longbow.jpg',
    'https://www.macmillandictionaryblog.com/wp-content/uploads/2017/08/sub-1024x548.jpg',
    'https://i.redd.it/7tie2v5psoh51.jpg',
]


def predict(url):
    print(url)
    resp = requests.post('http://localhost:8000/model', json={"image_url": url})
    print(resp.json())


if __name__ == '__main__':
    print('Making the first batch of predictions')
    for i in range(4):
        predict(cat_urls[i])
        predict(dog_urls[i])
        predict(other_urls[i])

    print('Labelling data and computing metrics, this can take a minute')
    compute_monitoring_metrics()

    print('Making the second batch of predictions')
    for i in range(4, len(cat_urls)):
        predict(cat_urls[i])
        predict(dog_urls[i])
        predict(other_urls[i])

    print('Labelling data and computing final metrics, this can take a minute')
    compute_monitoring_metrics()

    print('All done, check out metrics at http://localhost:8000/monitoring')
