1. install docker community

Install packages to allow apt to use a repository over HTTPS:

```sh
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
```

Add Docker’s official GPG key:
```sh
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

(following this official guide https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-using-the-repository)

to fix permission denied issue: https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket

to get around linux mint repo list: https://forums.linuxmint.com/viewtopic.php?t=300469

add it manually by editing the file /etc/apt/sources.list
manually add it to the sources.list and run sudo apt update everything will work
