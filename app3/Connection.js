import axios from 'axios'

class Connection {

    constructor(url, method) {
      this.url = url;
      this.method = method;
    }

    callService(){
        if(method == 'GET'){
            axios.get(url).then(resp => {
                return resp.data;
            });
        }
    }

  }


