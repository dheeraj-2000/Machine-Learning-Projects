#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Filename: hello-world.py
  """

from flask import Flask

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test():
    return 'Pinging Model Application!!'

if __name__ == ‘__main__’:
    app.run(debug=True, host=’0.0.0.0')

# @app.route('/users/<string:username>')
# def hello_world(username=None):

#     return("Hello {}!".format(username))


# In[ ]:




