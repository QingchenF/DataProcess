import os

from dingtalkchatbot.chatbot import DingtalkChatbot

webhook = 'https://oapi.dingtalk.com/robot/send?access_token=03d69526107ebfbb9124815833826ad2e3967307d9783bc23d1ba05ab340e6d3'
xiaoding = DingtalkChatbot(webhook)
Res = 'Warning:'
#xiaoding.send_text(msg=Res,is_at_all=False)
a = '''
curl 'https://oapi.dingtalk.com/robot/send?access_token=03d69526107ebfbb9124815833826ad2e3967307d9783bc23d1ba05ab340e6d3' \
-H 'Content-Type: application/json' \
-d '{"msgtype": "text",
    "text": {
         "content": "Warning:钉钉机器人群消息测试"
    }
  }'
'''
os.system('sh '+'curl.sh')