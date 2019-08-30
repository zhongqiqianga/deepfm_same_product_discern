import json
import requests
import base64



if __name__=="__main__":
       # client_id 为官网获取的AK， client_secret 为官网获取的SK
       host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=V9MHVocShfWIfZEqNUNEGGeP' \
              '&client_secret=tSNzOwWwXZ8lWg2Y7ESucl0Qgcn1ct6D'
       response = requests.get(host)
       content = response.json()
       access_token = content["access_token"]
       image = open(r'/Users/looker/Desktop/test.jpg', 'rb').read()
       data = {'image': base64.b64encode(image).decode()}
       request_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/classification/bolgger_photos_class" + "?access_token=" + access_token
       response = requests.post(request_url, data=json.dumps(data))
       content = response.json()
       print(content)

