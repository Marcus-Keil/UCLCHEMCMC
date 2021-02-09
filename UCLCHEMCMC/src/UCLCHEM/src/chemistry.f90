!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Chemistry module of UCL_CHEM.                                                               !
! Contains all the core machinery of the code, not really intended to be altered in standard  !
! use. Use a (custom) physics module to alter temp/density behaviour etc.                     !
!                                                                                             !
! chemistry module contains rates.f90, a series of subroutines to calculate all reaction rates!
! when updateChemistry is called from main, these rates are calculated, the ODEs are solved   !
! from currentTime to targetTime to get abundances at targetTime and then all abundances are  !
! written to the fullOutput file.                                                             !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE chemistry
USE physics
USE dvode_f90_m
USE network
USE photoreactions
USE surfacereactions
USE constants
IMPLICIT NONE
   !These integers store the array index of important species and reactions, x is for ions    
    INTEGER :: njunk,evapevents,ngrainco,readAbunds
    !loop counters    
    INTEGER :: i,j,l,writeStep,writeCounter=0

    !Flags to control desorption processes
    INTEGER :: h2desorb,crdesorb,uvdesorb,desorb,thermdesorb


    !Array to store reaction rates
    REAL(dp) :: rate(nreac)
    
    !Option column output
    character(LEN=15),ALLOCATABLE :: outSpecies(:)
    logical :: columnOutput=.False.,fullOutput=.False.
    INTEGER :: nout
    INTEGER, ALLOCATABLE :: outIndx(:)


    !DLSODE variables    
    INTEGER :: ITASK,ISTATE,NEQ,MXSTEP
    REAL(dp) :: reltol
    REAL(dp), ALLOCATABLE :: abstol(:)
    TYPE(VODE_OPTS) :: OPTIONS
    !initial fractional elemental abudances and arrays to store abundances
    REAL(dp) :: fh,fd,fhe,fc,fo,fn,fs,fmg,fsi,fcl,fp,ff,fli,fna,fpah,f15n,f13c,f18O
    REAL(dp) :: h2col,cocol,ccol
    REAL(dp),ALLOCATABLE :: abund(:,:),mantle(:)
    
    !Variables controlling chemistry
    REAL(dp) :: radfield,zeta,fr,omega,grainArea,cion,h2form,h2dis,lastTemp=0.0
    REAL(dp) :: ebmaxh2,epsilon,ebmaxcrf,ebmaxcr,phi,ebmaxuvcr,uv_yield,uvcreff
    REAL(dp), ALLOCATABLE ::vdiff(:)

    REAL(dp) :: turbVel=1.0


CONTAINS
!This gets called immediately by main so put anything here that you want to happen before the time loop begins, reader is necessary.
    SUBROUTINE initializeChemistry
        NEQ=nspec+1
        IF (ALLOCATED(abund)) DEALLOCATE(abund,vdiff,mantle)
        ALLOCATE(abund(NEQ,points),vdiff(SIZE(grainList)))
        CALL fileSetup
        !if this is the first step of the first phase, set initial abundances
        !otherwise reader will fix it
        IF (readAbunds.eq.0) THEN
            !ensure abund is initially zero
            abund= 0.
            !As default, have half in molecular hydrogen and half in atomic hydrogen
            abund(nh2,:) = 0.5*(1.0e0-fh)
            abund(nh,:) = fh

            !some elements default to atoms     
            abund(nd,:)=fd
            abund(nhe,:) = fhe                       
            abund(no,:) = fo  
            abund(nn,:) = fn               
            abund(nmg,:) = fmg
            abund(np,:) = fp
            abund(nf,:) = ff
            abund(nna,:) = fna
            abund(nli,:) = fli
            abund(npah,:) = fpah

            !others to ions
            abund(nsx,:) = fs
            abund(nsix,:) = fsi                
            abund(nclx,:) = fcl 
            
            !isotopes
            abund(n18o,:) = f18o  
            abund(n15n,:) = f15n           
            abund(n13c,:) = f13c    
            
            abund(nspec+1,:)=density      

            !Decide how much carbon is initiall ionized using parameters.f90
            SELECT CASE (ion)
                CASE(0)
                    abund(nc,:)=fc
                    abund(ncx,:)=1.d-10
                CASE(1)
                    abund(nc,:)=fc*0.5
                    abund(ncx,:)=fc*0.5
                CASE(2)
                    abund(nc,:)=1.d-10
                    abund(ncx,:)=fc
            END SELECT
            abund(nspec,:)=abund(ncx,:)+abund(nsix,:)+abund(nsx,:)+abund(nclx,:)

        ENDIF
        !Initial calculations of diffusion frequency for each species bound to grain
        !and other parameters required for diffusion reactions
        DO  i=lbound(grainList,1),ubound(grainList,1)
            j=grainList(i)
            vdiff(i)=VDIFF_PREFACTOR*bindingEnergy(i)/mass(j)
            vdiff(i)=dsqrt(vdiff(i))
        END DO

        !h2 formation rate initially set
        h2form = h2FormRate(gasTemp(dstep),dustTemp(dstep))
        ALLOCATE(mantle(points))
        DO l=1,points
            mantle(l)=sum(abund(grainList,l))
        END DO
        
        !DVODE SETTINGS
        ISTATE=1;;ITASK=1
        reltol=1e-4;MXSTEP=10000

        IF (.NOT. ALLOCATED(abstol)) THEN
            ALLOCATE(abstol(NEQ))
        END IF
        !OPTIONS = SET_OPTS(METHOD_FLAG=22, ABSERR_VECTOR=abstol, RELERR=reltol,USER_SUPPLIED_JACOBIAN=.FALSE.)
        

    END SUBROUTINE initializeChemistry

!Reads input reaction and species files as well as the final step of previous run if this is phase 2
    SUBROUTINE fileSetup
        IMPLICIT NONE
        integer i,j,l,m
        REAL(dp) junktemp

        INQUIRE(UNIT=11, OPENED=columnOutput)
        IF (columnOutput) write(11,333) specName(outIndx)
        333 format("Time,Density,gasTemp,av,",(999(A,:,',')))
        

        INQUIRE(UNIT=10, OPENED=fullOutput )
        IF (fullOutput) THEN
            write(10,334) fc,fo,fn,fs
            write(10,*) "Radfield ", radfield, " Zeta ",zeta
            write(10,335) specName
        END IF
        335 format("Time,Density,gasTemp,av,point,",(999(A,:,',')))
        334 format("Elemental abundances, C:",1pe15.5e3," O:",1pe15.5e3," N:",1pe15.5e3," S:",1pe15.5e3)


        !read start file if choosing to use abundances from previous run 
        !
        IF (readAbunds .eq. 1) THEN
            DO l=1,points
                READ(7,*) fhe,fc,fo,fn,fs,fmg
                READ(7,*) abund(:nspec,l)
                REWIND(7)
                abund(nspec+1,l)=density(l)
            END DO

            7010 format((999(1pe15.5,:,',')))    
        END IF
    END SUBROUTINE fileSetup

!Writes physical variables and fractional abundances to output file, called every time step.
    SUBROUTINE output

        IF (fullOutput) THEN
            write(10,8020) timeInYears,density(dstep),gasTemp(dstep),av(dstep),dstep,abund(:neq-1,dstep)
            8020 format(1pe11.3,',',1pe11.4,',',0pf8.2,',',1pe11.4,',',I4,',',(999(1pe15.5,:,',')))
        END IF

        !If this is the last time step of phase I, write a start file for phase II
        IF (readAbunds .eq. 0) THEN
           IF (switch .eq. 0 .and. timeInYears .ge. finalTime& 
               &.or. switch .eq. 1 .and.density(dstep) .ge. finalDens) THEN
               write(7,*) fhe,fc,fo,fn,fs,fmg
               write(7,8010) abund(:neq-1,dstep)
           ENDIF
        ENDIF
        8010  format((999(1pe15.5,:,',')))
        

        !Every 'writestep' timesteps, write the chosen species out to separate file
        !choose species you're interested in by looking at parameters.f90
        IF (writeCounter==writeStep .and. columnOutput) THEN
            writeCounter=1
            write(11,8030) timeInYears,density(dstep),gasTemp(dstep),av(dstep),abund(outIndx,dstep)
            8030  format(1pe11.3,',',1pe11.4,',',0pf8.2,',',1pe11.4,',',(999(1pe15.5,:,',')))
        ELSE
            writeCounter=writeCounter+1
        END IF
    END SUBROUTINE output

    SUBROUTINE updateChemistry
    !Called every time/depth step and updates the abundances of all the species
        !allow option for dens to have been changed elsewhere.
        IF (collapse .ne. 1) abund(nspec+1,dstep)=density(dstep)
        !y is at final value of previous depth iteration so set to initial values of this depth with abund
        !reset other variables for good measure        
        h2form = h2FormRate(gasTemp(dstep),dustTemp(dstep))
    
        !Sum of abundaces of all mantle species. mantleindx stores the indices of mantle species.
        mantle(dstep)=sum(abund(grainList,dstep))
        !evaluate co and h2 column densities for use in rate calculations
        !sum column densities of each point up to dstep. boxlength and dens are pulled out of the sum as common factors  
        IF (dstep.gt.1) THEN
            h2col=(sum(abund(nh2,:dstep-1)*density(:dstep-1))+0.5*abund(nh2,dstep)*density(dstep))*(cloudSize/real(points))
            cocol=(sum(abund(nco,:dstep-1)*density(:dstep-1))+0.5*abund(nco,dstep)*density(dstep))*(cloudSize/real(points))
            ccol=(sum(abund(nc,:dstep-1)*density(:dstep-1))+0.5*abund(nc,dstep)*density(dstep))*(cloudSize/real(points))

        ELSE
            h2col=0.5*abund(nh2,dstep)*density(dstep)*(cloudSize/real(points))
            cocol=0.5*abund(nco,dstep)*density(dstep)*(cloudSize/real(points))
            ccol=0.5*abund(nc,dstep)*density(dstep)*(cloudSize/real(points))
        ENDIF

        !call the actual ODE integrator
        CALL integrate

        !1.d-30 stops numbers getting too small for fortran.
        WHERE(abund<1.0d-30) abund=1.0d-30
        density(dstep)=abund(NEQ,dstep)
    END SUBROUTINE updateChemistry

    SUBROUTINE integrate
    !This subroutine calls DVODE (3rd party ODE solver) until it can reach targetTime with acceptable errors (reltol/abstol)
        DO WHILE(currentTime .lt. targetTime)         
            !reset parameters for DVODE
            ITASK=1 !try to integrate to targetTime
            ISTATE=1 !pretend every step is the first
            reltol=1e-4 !relative tolerance effectively sets decimal place accuracy
            abstol=1.0d-14*abund(:,dstep) !absolute tolerances depend on value of abundance
            WHERE(abstol<1d-30) abstol=1d-30 ! to a minimum degree

            !get reaction rates for this iteration
            CALL calculateReactionRates
            !Call the integrator.
            OPTIONS = SET_OPTS(METHOD_FLAG=22, ABSERR_VECTOR=abstol, RELERR=reltol,USER_SUPPLIED_JACOBIAN=.FALSE.,MXSTEP=MXSTEP)
            CALL DVODE_F90(F,NEQ,abund(:,dstep),currentTime,targetTime,ITASK,ISTATE,OPTIONS)
            SELECT CASE(ISTATE)
                CASE(-1)
                    !More steps required for this problem
                    MXSTEP=MXSTEP*2    
                CASE(-2)
                    !Tolerances are too small for machine but succesful to current currentTime
                    abstol=abstol*10.0
                CASE(-3)
                    write(*,*) "DVODE found invalid inputs"
                    write(*,*) "abstol:"
                    write(*,*) abstol
                    STOP
                CASE(-4)
                    !Successful as far as currentTime but many errors.
                    !Make targetTime smaller and just go again
                    targetTime=currentTime+10.0*SECONDS_PER_YEAR
                CASE(-5)
                    targetTime=currentTime*1.01
            END SELECT
        END DO                   
    END SUBROUTINE integrate

    !This is where reacrates subroutine is hidden
    include 'rates.f90'

    SUBROUTINE  F (NEQ, T, Y, YDOT)
        INTEGER, PARAMETER :: WP = KIND(1.0D0)
        INTEGER NEQ
        REAL(WP) T
        REAL(WP), DIMENSION(NEQ) :: Y, YDOT
        INTENT(IN)  :: NEQ, T, Y
        INTENT(OUT) :: YDOT
        REAL(dp) :: D,loss,prod
        !Set D to the gas density for use in the ODEs
        D=y(NEQ)
        ydot=0.0
        !The ODEs created by MakeRates go here, they are essentially sums of terms that look like k(1,2)*y(1)*y(2)*dens. Each species ODE is made up
        !of the reactions between it and every other species it reacts with.
        INCLUDE 'odes.f90'

        !H2 formation should occur at both steps - however note that here there is no 
        !temperature dependence. y(nh) is hydrogen fractional abundance.
        ydot(nh)  = ydot(nh) - 2.0*( h2form*y(nh)*D - h2dis*y(nh2) )
        !                             h2 formation - h2-photodissociation
        ydot(nh2) = ydot(nh2) + h2form*y(nh)*D - h2dis*y(nh2)
        !                       h2 formation  - h2-photodissociation
        ! get density change from physics module to send to DLSODE
        IF (collapse .eq. 1) ydot(NEQ)=densdot(y(NEQ))
    END SUBROUTINE F


    SUBROUTINE debugout
        open(79,file='output/debuglog',status='unknown')       !debug file.
        write(79,*) "Integrator failed, printing relevant debugging information"
        write(79,*) "dens",density(dstep)
        write(79,*) "density in integration array",abund(nspec+1,dstep)
        write(79,*) "Av", av(dstep)
        write(79,*) "Mantle", mantle(dstep)
        write(79,*) "Temp", gasTemp(dstep)
        DO i=1,nreac
            if (rate(i) .ge. huge(i)) write(79,*) "Rate(",i,") is potentially infinite"
        END DO
    END SUBROUTINE debugout
END MODULE chemistry