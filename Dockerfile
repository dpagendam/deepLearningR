FROM rocker/rstudio
RUN apt-get update
RUN sudo -u rstudio Rscript -e "install.packages('keras')"

RUN apt-get --assume-yes install python3-pip python3-dev
RUN apt-get --assume-yes install build-essential cmake git unzip pkg-config libopenblas-dev liblapack-dev
RUN apt-get --assume-yes install libhdf5-serial-dev
RUN pip3 install numpy scipy matplotlib tensorflow keras h5py

RUN sudo apt-get --assume-yes install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev libfftw3-dev libtiff5-dev libpng-dev
RUN sudo -u rstudio Rscript -e "install.packages('devtools')"
RUN sudo -u rstudio Rscript -e "library(devtools); install_github('dpagendam/deepLearningR')"

CMD ["/init"]

