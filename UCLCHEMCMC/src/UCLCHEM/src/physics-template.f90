!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!This is not a real module. It contains all the necessary variables and subroutines that  !
!are referenced or called by the rest of UCLCHEM. Any user wishing to create their own    !
!physics module should start from this template and ensure all the subroutines are present!
!                                                                                         !
!Our convention is that phase=1 should give the same results for all physics modules and  !
!the actual physics the module is trying to capture is performed when phase=2             !
!this allows the standard 2 phase run present in most UCLCHEM publications to be run from !
!one module.                                                                              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

MODULE physics
    USE constants
    USE network
    IMPLICIT NONE
    integer :: dstep,points
    !Switches for processes are also here, 1 is on/0 is off.
    integer :: collapse,switch,first,phase

    !evap changes evaporation mode (see chem_evaporate), ion sets c/cx ratio (see initializeChemistry)
    !Flags let physics module control when evap takes place.flag=0/1/2 corresponding to not yet/evaporate/done
    integer :: instantSublimation,ion,solidflag,volcflag,coflag,tempindx
    
    !variables either controlled by physics or that user may wish to change    
    double precision :: initialDens,dens,timeInYears,targetTime,currentTime,currentTimeold,finalDens,finalTime,grainRadius,initialTemp
    double precision :: cloudSize,rout,rin,baseAv,bc,olddens,maxTemp,vs
    double precision, allocatable :: av(:),coldens(:),temp(:)



CONTAINS
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !This is called once at the start of each UCLCHEM run. Allocate any arrays and set !
    !initial values for variables.                                                     !
    !If using python wrap, variables are not reset after the first run and therefore   !
    !must be reset here.                                                               !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    SUBROUTINE initializePhysics
    !Any initialisation logic steps go here
    !cloudSize is important as is allocating space for depth arrays
        allocate(av(points),coldens(points))
        cloudSize=(rout-rin)*pc
        if (collapse .eq. 1) THEN
            density=1.001*initialDens
        ELSE
            density=initialDens
        ENDIF 
    END SUBROUTINE

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Called every time loop in main.f90. Sets the timestep for the next output from   !
    !UCLCHEM. This is also given to the integrator as the targetTime in chemistry.f90 !
    !but the integrator itself chooses an integration timestep.                       !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    SUBROUTINE updateTargetTime
    IF (timeInYears .gt. 1.0d6) THEN
        targetTime=(timeInYears+1.0d5)*SECONDS_PER_YEAR
    ELSE IF (timeInYears .gt. 10000) THEN
        targetTime=(timeInYears+1000.0)*SECONDS_PER_YEAR
    ELSE IF (timeInYears .gt. 1000) THEN
        targetTime=(timeInYears+100.0)*SECONDS_PER_YEAR
    ELSE IF (timeInYears .gt. 0.0) THEN
        targetTime=(timeInYears*10)*SECONDS_PER_YEAR
    ELSE
        targetTime=3.16d7*10.d-8
    ENDIF
    END SUBROUTINE updateTargetTime



    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !This routine is called for every depth and time points and is the core of the module         !
    !                                                                                             !
    !Here, we should update the density (unless it is integrated, see densdot), temperature and   !
    !visual extinction at a minimum.                                                              !
    !Any other things that should be done each time step can be called from here                  !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    SUBROUTINE updatePhysics
        !calculate column density. Remember dstep counts from core to edge
        !and coldens should be amount of gas from edge to parcel.
        coldens(dstep)= cloudSize*((real(points-dstep))/real(points))*dens
        !calculate the Av using an assumed extinction outside of core (baseAv), depth of point and density
        av(dstep)= baseAv +coldens(dstep)/1.6d21

        !May wish to set density and temperature according to analytic functions or input from other model
    END SUBROUTINE updatePhysics




    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !This routine should check some logic for whether sublimation should occur                    !
    !                                                                                             !
    !For example, sputtering or thermal evaporation due to temperature increase                   !
    !It should then use grainList and gasGrainList from network module and the input abund array  !
    !to move material from the grain to gas phase.                                                !
    !                                                                                             !
    !See cloud.f90 for thermal example and cshock.f90 for sputtering                              !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    SUBROUTINE sublimation(abund)
        DOUBLE PRECISION :: abund(nspec+1,points)
        INTENT(INOUT) :: abund

    END SUBROUTINE sublimation


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !UCLCHEM/DVODE can integrate the density with the chemical abundances if ydot is  !
    !provided. If collapse=1, chemistry.f90 will call densdot to get the rate of      !
    !change of the density. Below is freefall example from Rawlings et al. 1992       !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    pure FUNCTION densdot()
    !Required for collapse=1, works out the time derivative of the density, allowing DVODE
    !to update density with the rest of our ODEs
    !It get's called by F, the SUBROUTINE in chem.f90 that sets up the ODEs for DVODE
        double precision :: densdot
        !Rawlings et al. 1992 freefall collapse. With factor bc for B-field etc
        IF (dens .lt. finalDens) THEN
             densdot=bc*(dens**4./initialDens)**0.33*&
             &(8.4d-30*initialDens*((dens/initialDens)**0.33-1.))**0.5
        ELSE
            densdot=1.0d-30       
        ENDIF    
    END FUNCTION densdot

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !As long as the above subroutines and variables are provided, anything else can be !
    !added to this module.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

END MODULE physics
