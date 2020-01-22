#!/usr/bin/env bash

function do_install_for_linux {
    echo "Installing sumo (used for traffic simulation and road network)"
    sudo add-apt-repository ppa:sumo/stable
    sudo apt-get update

    sudo apt-get install -y \
         python3-dev python3-venv python3-tk \
         libspatialindex-dev \
         sumo sumo-tools sumo-doc

    echo ""
    echo "-- dependencies have been installed --"
    echo ""
    echo "You'll need to set the SUMO_HOME variable. Logging out and back in will"
    echo "get you set up. Alternatively, in your current session, you can run:"
    echo ""
    echo "  source /etc/profile.d/sumo.sh"
    echo ""
}

function do_install_for_macos {
    echo "Installing sumo (used for traffic simulation and road network)"
    brew tap dlr-ts/sumo
    brew install sumo spatialindex

    # start X11 manually the first time, logging im/out will also do the trick
    open -g -a XQuartz.app

    echo ""
    echo "-- dependencies have been installed --"
    echo ""
    read -p "Add SUMO_HOME to ~/.bash_profile? [Yn]" should_add_SUMO_HOME
    echo "should_add_SUMO_HOME $should_add_SUMO_HOME"
    if [[ $should_add_SUMO_HOME =~ ^[yY\w]*$ ]]; then
        echo 'export SUMO_HOME="/usr/local/opt/sumo/share/sumo"' >> ~/.bash_profile
        echo "We've updated your ~/.bash_profile. Be sure to run:"
        echo ""
        echo "  source ~/.bash_profile"
        echo ""
        echo "in order to set the SUMO_HOME variable in your current session"
    else
        echo "Not updating ~/.bash_profile"
        echo "Make sure SUMO_HOME is set before proceeding"
    fi
}

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    echo "Detected Linux"
    do_install_for_linux
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    do_install_for_macos
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi
