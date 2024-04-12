import urllib3
from bs4 import BeautifulSoup
import io
from PIL import Image

http = urllib3.PoolManager()
i=0

for k in range(30000):
    if k % 500 == 0:
        print('Iteration number: ', k)
    try:
        url_coloured = 'https://m.ww2db.com/image.php?image_id={y}&ai_code=20230224'.format(y=k)
        response_coloured = http.request('GET', url_coloured)
    except Exception as e:
        pass

    # step 2: get url of jpg image from beautifulsoup object
    soup_col = BeautifulSoup(response_coloured.data, features="html.parser")

    # make sure we're looking at the colorized images - can identify using the below string
    if soup_col.get_text().find('Show WW2DB Colorized Version') != -1:
        a = soup_col.find_all("link")
        b = a[4].find_all("meta")
        im_url_col = b[-1]['content']
        im_url_bnw = im_url_col[:25] + im_url_col[37:]

        im_resp_col = http.request('GET', im_url_col)
        im_resp_bnw = http.request('GET', im_url_bnw)

        if im_resp_col.status == 404:
            pass
        else:
            resized_image_col = Image.open(io.BytesIO(im_resp_col.data))
            resized_image_bnw = Image.open(io.BytesIO(im_resp_bnw.data))

            resized_image_col.save('./ECE285/project/coloured/{x}.jpg'.format(x=i))
            resized_image_bnw.save('./ECE285/project/grayscale/{x}.jpg'.format(x=i))

            print('Image # {x} saved'.format(x=i))
            i+=1
