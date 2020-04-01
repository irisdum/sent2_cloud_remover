cd .
mkdir snap

cd snap
git clone https://github.com/senbox-org/snap-engine.git
cd snap-engine
mvn clean   install

cd ..
git clone https://github.com/senbox-org/snap-desktop.git
cd snap-desktop
mvn clean  install

