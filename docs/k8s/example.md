---
title: k8s:示例
comments: true
---

## Mongo Express + Mono DB

架构: web request ---> mongo express external service ---> mongo express pod --(使用 configmap, secret 配置)--> mongo db internal service ---> mongo db pod

### Secret

创建 mongo-secret.yaml 文件:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mongodb-secret
type: Opaque # 默认的键值对的形式
data:
  mongo_root_username: d2VuemV4dQ== # wenzexu的base64编码
  mongo_root_password: cGFzc3dvcmQ= # passwordd的base64编码
```

```
$ kubectl apply -f mongo-secret.yaml
```

### Mongo DB Deployment + Mongo DB Internal Service

创建 monodb.yaml 文件:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mongodb-depl
  labels:
    app: mongodb-depl
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mongodb
  template:
    metadata:
      labels:
        app: mongodb
    spec:
      containers:
        - name: mongodb
          image: mongo
          env:
            - name: MONGO_INITDB_ROOT_USERNAME
              valueFrom:
                secretKeyRef:
                  name: mongodb-secret
                  key: mongo_root_username
            - name: MONGO_INITDB_ROOT_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mongodb-secret
                  key: mongo_root_password
        #   ports:
        #   - containerPort: 27017 # 可以省略

--- # 注意, 两个YAML文件可以合并为一个, 中间要有---分隔符

apiVersion: v1
kind: Service
metadata:
  name: mongodb-service
spec:
  selector:
    app: mongodb
  ports:
    - protocol: TCP
      port: 27017
      targetPort: 27017 # 可以省略, 因为和port一样
```

```
$ kubectl apply -f mongodb.yaml
$ kubectl get pod -o wide
$ kubectl describe service mongodb-service
$ kubectl get all | grep mongodb
```

### Configmap

创建 mongo-configmap.yaml 文件:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mongodb-configmap
data:
  database_url: mongodb-service
```

```
$ kubectl apply -f mongo-configmap.yaml
```

### Mongo Express Deployment + Mongo Express External Service

创建 mongoexpress.yaml 文件:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mongoexpress-depl
  labels:
    app: mongoexpress-depl
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mongoexpress
  template:
    metadata:
      labels:
        app: mongoexpress
    spec:
      containers:
        - name: mongoexpress
          image: mongo-express
          env:
          - name: ME_CONFIG_MONGODB_ADMINUSERNAME
            valueFrom:
              secretKeyRef:
                name: mongodb-secret
                key: mongo_root_username
          - name: ME_CONFIG_MONGODB_ADMINPASSWORD
            valueFrom:
              secretKeyRef:
                name: mongodb-secret
                key: mongo_root_password
          - name: ME_CONFIG_MONGODB_SERVER
            valueFrom:
              configMapKeyRef:
                name: mongodb-configmap
                key: database_url
          - name: ME_CONFIG_MONGODB_AUTH_DATABASE
            value: admin
          # ports:
          # - containerPort: 8081

---

apiVersion: v1
kind: Service
metadata:
  name: mongoexpress-service
spec:
  type: LoadBalancer # k8s的团队脑子有问题, 这个LoadBalancer其实是external service的意思, 因为就算是internal service也会有负载均衡的功能. 定义这个会给service分配一个外部的ip地址, 用来外部访问
  selector:
    app: mongoexpress
  ports:
    - protocol: TCP
      port: 8081
      targetPort: 8081 # 可以省略, 因为和port一样
      nodePort: 30000 # 必须在30000-32767, 这是k8s给service分配的外部的ip地址的端口
```

```
$ kubectl get service
NAME                   TYPE           CLUSTER-IP        EXTERNAL-IP    PORT(S)          AGE
kubernetes             ClusterIP      192.168.194.129   <none>         443/TCP          4h3m
mongodb-service        ClusterIP      192.168.194.171   <none>         27017/TCP        20m
mongoexpress-service   LoadBalancer   192.168.194.146   198.19.249.2   8081:30000/TCP   27s
```

如上述所示, external service获取到了一个内部的ip, `192.168.194.146`和一个外部的ip, `198.19.249.2`, 所以上述三个端口的含义为:

1. 流量进入`198.19.249.2:30000`
2. 流量被转发到`192.168.192.146:8081`
3. 流量经过负载均衡, 转发到`<pod1的内部ip>:8081`, `<pod2的内部ip>:8081`, ...

这个时候, 访问`198.19.249.2:30000`, 输入Mongo Express的默认用户名密码, admin:pass, 就可以看到Mongo Express的界面.