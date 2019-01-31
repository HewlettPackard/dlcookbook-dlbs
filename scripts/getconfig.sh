#!/bin/bash
#sudo_dir="/mops/linux"
sudo_dir="/admin/scripts"

echo "sysinfo"
sudo ${sudo_dir}/sysinfo/sysinfo
echo
echo

echo "lsb_release"
lsb_release -a
echo
echo

echo "uname -r"
uname -r
echo
echo

echo "os-release"
cat /etc/os-release
echo
echo

echo "hostnamectl"
hostnamectl
echo
echo

echo "NVidia GPU Information"
echo "nvidia-smi -q -i 0"
nvidia-smi -q -i 0
echo "nvidia-smi -L"
nvidia-smi -L
echo "nvidia-smi topo -m"
nvidia-smi topo -m

echo
echo
echo "PCI Devices (lspci)"
lspci



echo
echo
echo "/proc/cpuinfo"
cat /proc/cpuinfo

echo
echo
echo "/proc/meminfo"
cat /proc/meminfo

echo
echo
echo "Block devices lsblk"
lsblk

echo
echo
echo "list hardware lshw"
sudo lshw -short

echo
echo
echo "Network interfaces"
sudo lshw -class network -short

echo
echo
echo "Infiniband lspci"
lspci |grep ib

echo
echo
echo 'ip link show'
ip link show

echo
echo
echo 'lscpu'
lscpu

echo
echo
echo 'meminfo'
cat /proc/meminfo
