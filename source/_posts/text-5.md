---
title: 持续集成与持续部署(基础设施搭建）
date: 2020-03-12 09:21:44
tags: [项目建设, CI, CD]
categories: 产品养成
---

## 目标
构建一套以Jenkins为核心的自动测试与持续集成系统，帮助APP端和服务端进行自动化打包，测试，存储，发布等一系列高度重复的任务。

<!--more-->

流程: Github上的push操作触发webhook，提送到本地Mac使用ngork反向代理暴露出的Jenkins端口，Jenkins启动自动测试与打包过程，结果通过邮件通知对应开发者，打包完成的ipa，apk，jar等安装包按规则命名存放到指定位置，该位置可以通过本地的nginx服务在局域网内被访问，测试拿到安装包进行手工测试。

## 环境
为了满足iOS APP打包需要Mac环境，现在阿里云腾讯云也没有提供Mac OS环境的服务器，所以在本地准备一台Mac环境(例如最便宜的Mac Mini）,在Mac上部署环境。

## 安装Jenkins

---
以下是坑，如果你已经在坑里，可能会帮到你

从官网下载dmg安装Jenkins
[the Jenkins download page](https://jenkins.io/download)

安装后
拿到admin的密码
cat /var/log/jenkins/jenkins.log

由于dmg安装后默认建立一个Jenkins用户，无法直接使用本机安装的各项环境，所有需要
设置Jenkins权限为当前用户

``` shell
# 停止Jenkins
sudo launchctl unload /Library/LaunchDaemons/org.jenkins-ci.plist
```

```shell
# 修改Group和User
# <用户名>填写你的MacOS用户名，不知道的可以在命令行使用whoami查看，不需要尖括号
$ sudo vim +1 +/daemon +’s/daemon/staff/’ +/daemon +’s/daemon/<用户名> +wq org.jenkins-ci.plist
```
```shell
# 可能相应文件夹的权限
sudo chown -R <用户名>:staff /Users/Shared/Jenkins/
sudo chown -R <用户名>:staff /var/log/jenkins/
```

```shell
# 启动Jenkins
sudo launchctl load /Library/LaunchDaemons/org.jenkins-ci.plist
```

重启失败产尝试：
```shell
sudo /usr/bin/java -Dfile.encoding=UTF-8 -XX:PermSize=256m -XX:MaxPermSize=512m -Xms256m -Xmx512m -Djava.io.tmpdir=/Users/Shared/Jenkins/tmp -jar /Applications/Jenkins/jenkins.war --httpPort=8080 --enable-future-java
```
以上都是坑，如果你失败了，是正常的，请放弃dmg安装。

---

使用brew安装
```shell
brew install jenkins-lts
```
如果未安装brew，先安装brew
```shell
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)”
```

安装完成
```shell
To have launchd start jenkins now and restart at login:
  brew services start jenkins-lts
Or, if you don't want/need a background service you can just run:
  jenkins-lts
```
不要着急启动Jenkins，设置Jenkins的环境配置
先去/Library/LaunchDaemons目录下新建一个org.jenkins-ci.plist文件
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>StandardOutPath</key>
    <string>/var/log/jenkins/jenkins.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/jenkins/jenkins.log</string>
    <key>EnvironmentVariables</key>
    <dict>
      <key>JENKINS_HOME</key>
      <string>/Users/<你的目录>/Documents/FuckingJenkins/Jenkins/Home</string>
    </dict>
    <key>GroupName</key>
    <string>daemon</string>
    <key>KeepAlive</key>
    <true/>
    <key>Label</key>
    <string>org.jenkins-ci</string>
    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>/Library/Application Support/Jenkins/jenkins-runner.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>UserName</key>
    <string>jenkins</string>
    <key>SessionCreate</key>
    <true/>
  </dict>
</plist>
```

为避免权限问题，先执行下面的命令行“
```shell
sudo chown root /usr/local/Cellar/jenkins-lts/2.204/homebrew.mxcl.jenkins.plist
```

启动
```shell
brew services start jenkins-lts
```

局域网无法访问的问题：
修改 homebrew.mxcl.jenkins.plist 的  httpListenAddress 为 0.0.0.0
路径:
~/Library/LaunchAgents/homebrew.mxcl.jenkins.plist
/usr/local/Cellar/jenkins/版本号/homebrew.mxcl.jenkins.plist

重启Jenkins
```shell
brew services restart jenkins-lts
```

启动Jenkins后的设置不再赘述，以下再提几点常见问题

**git配置**
Jenkins-》Manage Jenkins-Global Tool Configuration-》Path to Git executable
配置 /usr/local/bin/git

本机git可能保存有默认的用户名密码，这会导致Jenkins在拉取git仓库的时候使用git默认的用户名密码而不是你配置的credit，如果出现git拉取秒挂的问题，可以检查以下这个。


**提供一个email notification的template**
```xml
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>${PROJECT_NAME}-第${BUILD_NUMBER}次构建日志</title>
</head>
 
<body leftmargin="8" marginwidth="0" topmargin="8" marginheight="4"
    offset="0">
    <table width="95%" cellpadding="0" cellspacing="0"
        style="font-size: 11pt; font-family: Tahoma, Arial, Helvetica, sans-serif">
        <tr>
            <td>(本邮件是程序自动下发的，请勿回复！)</td>
        </tr>
        <tr>
            <td><h2>
                    <font color="#0000FF">构建结果 - ${BUILD_STATUS}</font>
                </h2></td>
        </tr>
        <tr>
            <td><br />
            <b><font color="#0B610B">构建信息</font></b>
            <hr size="2" width="100%" align="center" /></td>
        </tr>
        <tr>
            <td>
                <ul>
                    <li>项目名称 ： ${PROJECT_NAME}</li>
                    <li>构建编号 ： 第${BUILD_NUMBER}次构建</li>
                    <li>触发原因： ${CAUSE}</li>
                    <li>构建日志： <a href="${BUILD_URL}console">${BUILD_URL}console</a></li>
                    <li>构建  Url ： <a href="${BUILD_URL}">${BUILD_URL}</a></li>
                    <li>工作目录 ： <a href="${PROJECT_URL}ws">${PROJECT_URL}ws</a></li>
                    <li>项目  Url ： <a href="${PROJECT_URL}">${PROJECT_URL}</a></li>
                </ul>
            </td>
        </tr>
        <tr>
            <td><b><font color="#0B610B">Changes Since Last
                        Successful Build:</font></b>
            <hr size="2" width="100%" align="center" /></td>
        </tr>
        <tr>
            <td>
                <ul>
                    <li>历史变更记录 : <a href="${PROJECT_URL}changes">${PROJECT_URL}changes</a></li>
                </ul> ${CHANGES_SINCE_LAST_SUCCESS,reverse=true, format="Changes for Build #%n:<br />%c<br />",showPaths=true,changesFormat="<pre>[%a]<br />%m</pre>",pathFormat="    %p"}
            </td>
        </tr>
        <tr>
            <td><b>Test Informations</b>
            <hr size="2" width="100%" align="center" /></td>
        </tr>
        <tr>
            <td><pre
                    style="font-size: 11pt; font-family: Tahoma, Arial, Helvetica, sans-serif">Total:${TEST_COUNTS,var="total"},Pass:${TEST_COUNTS,var="pass"},Failed:${TEST_COUNTS,var="fail"},Skiped:${TEST_COUNTS,var="skip"}</pre>
                <br /></td>
        </tr>
        <tr>
            <td><b><font color="#0B610B">构建日志 (最后 100行):</font></b>
            <hr size="2" width="100%" align="center" /></td>
        </tr>
        <tr>
            <td><textarea cols="80" rows="30" readonly="readonly"
                    style="font-family: Courier New">${BUILD_LOG, maxLines=100}</textarea>
            </td>
        </tr>
    </table>
</body>
</html>
```

## 安装ngrok
注册ngrok并获取token，注册后的ngrok将不再有时长限制，不然每次地址会有8小时超时时间
```shell
./ngrok authtoken 1Yexxxxxxxxxxxxx
# 将ngrok设置为后台运行
./nohup ./ngrok http 8080 -log=stdout &
```

## 安装fastlane

* 更新ruby
首先安装RVM
```shell
curl -L get.rvm.io | bash -s stable
rvm reinstall 2.6.3
```

* 环境设置
在Mac OS 10.14 版本以前
```shell
xcode-select --install 
```
在Mac OS 10.14 版本之后
```shell
open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
```
* 安装fastlane
```shell
sudo gem install -n /usr/local/bin fastlane --verbose
```

