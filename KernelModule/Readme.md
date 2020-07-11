//////////////Компиляция модуля ядра (вывод из консоли)
slava@slava-VirtualBox:~/KernelModule$ make
make -C /lib/modules/5.3.0-28-generic/build M=/home/slava/KernelModule modules
make[1]: вход в каталог «/usr/src/linux-headers-5.3.0-28-generic»
  CC [M]  /home/slava/KernelModule/example.o
  Building modules, stage 2.
  MODPOST 1 modules
  CC      /home/slava/KernelModule/example.mod.o
  LD [M]  /home/slava/KernelModule/example.ko
make[1]: выход из каталога «/usr/src/linux-headers-5.3.0-28-generic»
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$
//////////////Внедрение модуля в ядро и его последующее удаление 
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$
slava@slava-VirtualBox:~/KernelModule$ sudo insmod example.ko
[sudo] пароль для slava: 
slava@slava-VirtualBox:~/KernelModule$ sudo dmesg
[    0.000000] Linux version 5.3.0-28-generic (buildd@lcy01-amd64-009) (gcc version 7.4.0 (Ubuntu 7.4.0-1ubuntu1~18.04.1)) #30~18.04.1-Ubuntu SMP Fri Jan 17 06:14:09 UTC 2020 (Ubuntu 5.3.0-28.30~18.04.1-generic 5.3.13)

...
...
...

[ 3633.941087] example: module verification failed: signature and/or required key missing - tainting kernel
[ 3633.942315] Hello world, it's init module
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ lsmod | grep "example"
example                16384  0
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ sudo rmmod example
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ sudo dmesg
[    0.000000] Linux version 5.3.0-28-generic (buildd@lcy01-amd64-009) (gcc version 7.4.0 (Ubuntu 7.4.0-1ubuntu1~18.04.1)) #30~18.04.1-Ubuntu SMP Fri Jan 17 06:14:09 UTC 2020 (Ubuntu 5.3.0-28.30~18.04.1-generic 5.3.13)

...
...
...

[ 3821.980791] Goodbye world, it's cleanup module
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ 
slava@slava-VirtualBox:~/KernelModule$ lsmod | grep "example"
slava@slava-VirtualBox:~/KernelModule$ 
