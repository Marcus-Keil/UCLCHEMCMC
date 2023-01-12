!Marcus Keil Original 13/03/2020 Last Update 3/11/2020
!Python wrapper for uclchem, compiled with "make python"
!general becomes a python function which takes a dictionary of parameters
!and a string of delimited species names
MODULE wrap
    USE physics
    USE chemistry
    IMPLICIT NONE
    CHARACTER (LEN=100) :: abundFile, outputFile, columnFile, outFile

CONTAINS
    SUBROUTINE run_model_to_file(dictionary, outSpeciesIn)
        CHARACTER(LEN=*) :: dictionary, outSpeciesIn
        !f2py intent(in) dictionary,outSpeciesIn

        ! Set the boolean to True if you want to be able to return a python array
        ! call the dictionary_parser function in order to read the dictionary of parameters
        INCLUDE 'defaultparameters.f90'
        CALL dictionary_parser(dictionary, outSpeciesIn)

        CALL solveAbundances

        !close outputs to attempt to force flush
        close(10)
        close(11)
        close(7)
    END SUBROUTINE run_model_to_file

    SUBROUTINE run_model_for_abundances(dictionary, outSpeciesIn,abundance_out)
        CHARACTER(LEN=*) :: dictionary, outSpeciesIn
        DOUBLE PRECISION :: abundance_out(2)
        !f2py intent(in) dictionary,outSpeciesIn
        !f2py intent(out) abundance_out

        ! Set the boolean to True if you want to be able to return a python array
        ! call the dictionary_parser function in order to read the dictionary of parameters
        INCLUDE 'defaultparameters.f90'

        !Read input parameters from the dictionary
        CALL dictionary_parser(dictionary, outSpeciesIn)

        !close outputs to attempt to force flush
        close(10)
        close(11)
        close(7)

        CALL solveAbundances
        abundance_out=abund(outIndx,1)
    END SUBROUTINE run_model_for_abundances

    SUBROUTINE solveAbundances()
        CALL initializePhysics
        CALL initializeChemistry

        dstep=1
        currentTime=0.0
        timeInYears=0.0

        !loop until the end condition of the model is reached
        DO WHILE ((switch .eq. 1 .and. density(1) < finalDens) .or. (switch .eq. 0 .and. timeInYears < finalTime))
            !store current time as starting point for each depth step
            currentTimeold=currentTime

            !Each physics module has a subroutine to set the target time from the current time
            CALL updateTargetTime

            !loop over parcels, counting from centre out to edge of cloud
            DO dstep=1,points
                !update chemistry from currentTime to targetTime
                CALL updateChemistry

                currentTime=targetTime
                !get time in years for output, currentTime is now equal to targetTime
                timeInYears= currentTime/SECONDS_PER_YEAR

                !Update physics so it's correct for new currentTime and start of next time step
                CALL updatePhysics
                !Sublimation checks if Sublimation should happen this time step and does it
                CALL sublimation(abund)
                !write this depth step now time, chemistry and physics are consistent
                CALL output

                !reset time for next depth point
                if (points .gt. 1)currentTime=currentTimeold
            END DO
        END DO
    END SUBROUTINE solveAbundances



    SUBROUTINE dictionary_parser(dictionary, outSpeciesIn)
        CHARACTER(LEN=*) :: dictionary, outSpeciesIn
        INTEGER, ALLOCATABLE, DIMENSION(:) :: locations
        LOGICAL :: ChemicalDuplicateCheck
        INTEGER :: posStart, posEnd, whileInteger,inputindx
        CHARACTER(LEN=100) :: inputParameter, inputValue

        close(10)
        close(11)
        close(7)

        whileInteger = 0

        posStart = scan(dictionary, '{')

        DO WHILE (whileInteger .NE. 1)
            posEnd = scan(dictionary, ':')
            inputParameter = dictionary(posStart+2:posEnd-2)
            dictionary = dictionary(posEnd:)
            posStart = scan(dictionary, ' ')
            IF (scan(dictionary, ',') .EQ. 0) THEN
                posEnd = scan(dictionary, '}')
                whileInteger = 1
            ELSE
                posEnd = scan(dictionary, ',')
            END IF
            inputValue = dictionary(posStart+1:posEnd-1)


            SELECT CASE (inputParameter)
                CASE('alpha')
                    !To provide alphas, set keyword alpha in inputdictionary with a dictionary value
                    !that dictionary should be index:value pairs for the alpha array
                    posStart=scan(dictionary,'{')
                    posEnd=scan(dictionary,'}')
                    CALL alpha_parser(dictionary(posStart+1:posEnd))
                CASE('initialTemp')
                    READ(inputValue,*) initialTemp
                CASE('maxTemp')
                    READ(inputValue,*) maxTemp
                CASE('initialDens')
                    READ(inputValue,*) initialDens
                CASE('finalDens')
                    READ(inputValue,*) finalDens
                CASE('currentTime')
                    READ(inputValue,*) currentTime
                CASE('finalTime')
                    READ(inputValue,*) finalTime
                CASE('radfield')
                    READ(inputValue,*) radfield
                CASE('zeta')
                    READ(inputValue,*) zeta
                CASE('fr')
                    READ(inputValue,*) fr
                CASE('rout')
                    READ(inputValue,*) rout
                CASE('rin')
                    READ(inputValue,*) rin
                CASE('baseAv')
                    READ(inputValue,*) baseAv
                CASE('points')
                    READ(inputValue,*) points
                CASE('switch')
                    Read(inputValue,*) switch
                CASE('collapse')
                    READ(inputValue,*) collapse
                CASE('bc')
                    READ(inputValue,*) bc
                CASE('readAbunds')
                    READ(inputValue,*) readAbunds
                CASE('phase')
                    READ(inputValue,*) phase
                CASE('desorb')
                    READ(inputValue,*) desorb
                CASE('h2desorb')
                    READ(inputValue,*) h2desorb
                CASE('crdesorb')
                    READ(inputValue,*) crdesorb
                CASE('uvdesorb')
                    READ(inputValue,*) uvdesorb
                CASE('instantSublimation')
                    READ(inputValue,*) instantSublimation
                CASE('ion')
                    READ(inputValue,*) ion
                CASE('tempindx')
                    READ(inputValue,*) tempindx
                CASE('fhe')
                    READ(inputValue,*) fhe
                CASE('fc')
                    READ(inputValue,*) fc
                CASE('fo')
                    READ(inputValue,*) fo
                CASE('fn')
                    READ(inputValue,*) fn
                CASE('fs')
                    READ(inputValue,*) fs
                CASE('fmg')
                    READ(inputValue,*) fmg
                CASE('fsi')
                    READ(inputValue,*) fsi
                CASE('fcl')
                    READ(inputValue,*) fcl
                CASE('fp')
                    READ(inputValue,*) fp
                CASE('ff')
                    READ(inputValue,*) ff
                CASE('outSpecies')
                    IF (ALLOCATED(outIndx)) DEALLOCATE(outIndx)
                    IF (ALLOCATED(outSpecies)) DEALLOCATE(outSpecies)
                    READ(inputValue,*) nout
                    ALLOCATE(outIndx(nout))
                    ALLOCATE(outSpecies(nout))
                    IF (outSpeciesIn .eq. "") THEN
                        write(*,*) "Outspecies parameter set but no outspecies string given"
                        write(*,*) "general(parameter_dict,outSpeciesIn) requires a delimited string of species names"
                        write(*,*) "if outSpecies or columnFlag is set in the parameter dictionary"
                        STOP
                    ELSE
                        READ(outSpeciesIn,*, END=22) outSpecies
                        IF (outSpeciesIn .eq. "") THEN
22                              write(*,*) "mismatch between outSpeciesIn and number given in dictionary"
                            write(*,*) "Number:",nout
                            write(*,*) "Species list:",outSpeciesIn
                            STOP
                        END IF
                    END IF
                    !assign array indices for important species to the integers used to store them.
                    DO i=1,nspec
                        DO j=1,nout
                            IF (specname(i).eq.outSpecies(j)) outIndx(j)=i
                        END DO
                    END DO
                CASE('writeStep')
                    READ(inputValue,*) writeStep
                CASE('ebmaxh2')
                    READ(inputValue,*) ebmaxh2
                CASE('epsilon')
                    READ(inputValue,*) epsilon
                CASE('ebmaxcrf')
                    READ(inputValue,*) ebmaxcrf
                CASE('uvcreff')
                    READ(inputValue,*) uvcreff
                CASE('ebmaxcr')
                    READ(inputValue,*) ebmaxcr
                CASE('phi')
                    READ(inputValue,*) phi
                CASE('ebmaxuvcr')
                    READ(inputValue,*) ebmaxuvcr
                CASE('uv_yield')
                    READ(inputValue,*) uv_yield
                CASE('omega')
                    READ(inputValue,*) omega
                CASE('vs')
                    READ(inputValue,*) vs
                CASE('abundFile')
                    READ(inputValue,*) abundFile
                    abundFile = trim(abundFile)
                    open(7,file=abundFile,status='unknown')
                CASE('outputFile')
                    READ(inputValue,*) outFile
                    outputFile = trim(outFile)
                    open(10,file=outputFile,status='unknown')
                CASE('columnFile')
                    IF (trim(outSpeciesIn) .NE. '' ) THEN
                        READ(inputValue,*) columnFile
                        columnFile = trim(columnFile)
                        open(11,file=columnFile,status='unknown')
                    ELSE
                        WRITE(*,*) "Error in output species. No species were given but a column file was given."
                        WRITE(*,*) "columnated output requires output species to be chosen."
                        STOP
                    END IF

                CASE DEFAULT
                    WRITE(*,*) "Problem with given parameter: '", trim(inputParameter), "'."
                    WRITE(*,*) "This is either not supported yet, or invalid."
            END SELECT
            dictionary = dictionary(posEnd:)
            IF (SCAN(dictionary,',') .eq. 0) whileInteger=1
        END DO
    END SUBROUTINE dictionary_parser

    SUBROUTINE alpha_parser(alpha_string)
        INTEGER :: inputIndx,posStart,posEnd
        CHARACTER(LEN=100) :: inputValue
        CHARACTER(LEN=*) :: alpha_string
        LOGICAL :: continue_flag

        continue_flag=.True.
        DO WHILE (continue_flag)
            !substring containing integer key
            posStart=1
            posEnd=SCAN(alpha_string,':')
            !read it into index integer
            READ(alpha_string(posStart:posEnd-1),*) inputindx

            !substring including alpha value for the index.
            posStart=posEnd+1
            posEnd=SCAN(alpha_string,',')
            !last value will have a } instead of , so grab index and tell loop to finish
            IF (posEnd .eq. 0) THEN
                posEnd=SCAN(alpha_string,"}")
                continue_flag=.False.
            END IF

            !read that substring
            inputValue=alpha_string(posStart:posEnd-1)
            READ(inputValue,*) alpha(inputIndx)
            !update string to remove this entry
            alpha_string=alpha_string(posEnd+1:)
        END DO
    END SUBROUTINE alpha_parser

    SUBROUTINE TO_DF(dictionary, outSpeciesIn, numberPoints, parameterArray, chemicalAbunArray, allChemicalAbunArray, stepCount)
        USE physics
        USE chemistry
        CHARACTER(LEN=*) :: dictionary, outSpeciesIn
        CHARACTER(LEN=LEN(outSpeciesIn)) :: tempOutSpecies
        INTEGER :: posStart, posEnd, whileInteger, numberPoints, chemStart, chemEnd
        CHARACTER (LEN=100) :: inputParameter, inputValue
        CHARACTER (LEN=100) :: abundFile, outputFile, columnFile
        INTEGER :: itterationInteger
        INTEGER*4, INTENT(INOUT) :: stepCount(1)
        !f2py intent(in out) stepCount
        DOUBLE PRECISION, INTENT(INOUT), DIMENSION(10000, numberPoints, 12) :: parameterArray
        DOUBLE PRECISION, INTENT(INOUT), DIMENSION(10000, numberPoints, 215) :: chemicalAbunArray
        DOUBLE PRECISION, INTENT(INOUT), DIMENSION(10000, numberPoints, 215) :: allChemicalAbunArray
        !f2py intent(in out) parameterArray
        !f2py depend(numberPoints) parameterArray
        !f2py intent(in out) chemicalAbunArray
        !f2py depend(numberPoints) chemicalAbunArray
        !f2py intent(in out) allChemicalAbunArray
        !f2py depend(numberPoints) allChemicalAbunArray
        CHARACTER(LEN=10), ALLOCATABLE, DIMENSION(:) :: chemicals
        CHARACTER(LEN=10) :: finalOnly = "False"
        LOGICAL :: readAbundsBool = .False.
        DOUBLE PRECISION, DIMENSION(numberPoints) :: tempAV
        DOUBLE PRECISION :: temph2form, tempfc, tempfo, tempfmg, tempfhe, tempdstep
        include 'defaultparameters.f90'
        close(10)
        close(11)
        close(7)

        IF (scan(dictionary, 'columnFile') .EQ. 0) THEN
            columnFlag=.False.
        END IF

        whileInteger = 0

        posStart = scan(dictionary, '{')

        DO WHILE (whileInteger .NE. 1)
            posEnd = scan(dictionary, ':')
            inputParameter = dictionary(posStart+2:posEnd-2)
            dictionary = dictionary(posEnd:)
            posStart = scan(dictionary, ' ')
            IF (scan(dictionary, ',') .EQ. 0) THEN
                posEnd = scan(dictionary, '}')
                whileInteger = 1
            ELSE
                posEnd = scan(dictionary, ',')
            END IF
            inputValue = dictionary(posStart+1:posEnd-1)
            dictionary = dictionary(posEnd:)

            SELECT CASE (inputParameter)
                CASE('alpha')
                    !To provide alphas, set keyword alpha in inputdictionary with a dictionary value
                    !that dictionary should be index:value pairs for the alpha array
                    posStart=scan(dictionary,'{')
                    posEnd=scan(dictionary,'}')
                    CALL alpha_parser(dictionary(posStart+1:posEnd))
                CASE('initialTemp')
                    READ(inputValue,*) initialTemp
                CASE('maxTemp')
                    READ(inputValue,*) maxTemp
                CASE('initialDens')
                    READ(inputValue,*) initialDens
                CASE('finalDens')
                    READ(inputValue,*) finalDens
                CASE('currentTime')
                    READ(inputValue,*) currentTime
                CASE('finalTime')
                    READ(inputValue,*) finalTime
                CASE('radfield')
                    READ(inputValue,*) radfield
                CASE('zeta')
                    READ(inputValue,*) zeta
                CASE('fr')
                    READ(inputValue,*) fr
                CASE('rout')
                    READ(inputValue,*) rout
                CASE('rin')
                    READ(inputValue,*) rin
                CASE('baseAv')
                    READ(inputValue,*) baseAv
                CASE('points')
                    READ(inputValue,*) points
                CASE('switch')
                    Read(inputValue,*) switch
                CASE('collapse')
                    READ(inputValue,*) collapse
                CASE('bc')
                    READ(inputValue,*) bc
                CASE('readAbunds')
                    READ(inputValue,*) readAbunds
                CASE('phase')
                    READ(inputValue,*) phase
                CASE('desorb')
                    READ(inputValue,*) desorb
                CASE('h2desorb')
                    READ(inputValue,*) h2desorb
                CASE('crdesorb')
                    READ(inputValue,*) crdesorb
                CASE('uvdesorb')
                    READ(inputValue,*) uvdesorb
                CASE('instantSublimation')
                    READ(inputValue,*) instantSublimation
                CASE('ion')
                    READ(inputValue,*) ion
                CASE('tempindx')
                    READ(inputValue,*) tempindx
                CASE('fhe')
                    READ(inputValue,*) fhe
                CASE('fc')
                    READ(inputValue,*) fc
                CASE('fo')
                    READ(inputValue,*) fo
                CASE('fn')
                    READ(inputValue,*) fn
                CASE('fs')
                    READ(inputValue,*) fs
                CASE('fmg')
                    READ(inputValue,*) fmg
                CASE('fsi')
                    READ(inputValue,*) fsi
                CASE('fcl')
                    READ(inputValue,*) fcl
                CASE('fp')
                    READ(inputValue,*) fp
                CASE('ff')
                    READ(inputValue,*) ff
                CASE('outSpecies')
                    IF (ALLOCATED(outIndx)) DEALLOCATE(outIndx)
                    READ(inputValue,*) nout
                    ALLOCATE(outIndx(nout))
                    ALLOCATE(chemicals(nout))
                    chemEnd = 0
                    tempOutSpecies = outSpeciesIn
                    DO j=1, nout
                        tempOutSpecies = tempOutSpecies(chemEnd+1:)
                        chemEnd = scan(tempOutSpecies, ' ')
                        chemicals(j) = TRIM(tempOutSpecies(:chemEnd-1))
                    END DO
                    IF (ALLOCATED(outSpecies)) DEALLOCATE(outSpecies)
                    outSpecies = chemicals
                CASE('writeStep')
                    READ(inputValue,*) writeStep
                CASE('ebmaxh2')
                    READ(inputValue,*) ebmaxh2
                CASE('epsilon')
                    READ(inputValue,*) epsilon
                CASE('ebmaxcrf')
                    READ(inputValue,*) ebmaxcrf
                CASE('uvcreff')
                    READ(inputValue,*) uvcreff
                CASE('ebmaxcr')
                    READ(inputValue,*) ebmaxcr
                CASE('phi')
                    READ(inputValue,*) phi
                CASE('ebmaxuvcr')
                    READ(inputValue,*) ebmaxuvcr
                CASE('uv_yield')
                    READ(inputValue,*) uv_yield
                CASE('omega')
                    READ(inputValue,*) omega
                CASE('vs')
                    READ(inputValue,*) vs
                CASE('abundFile')
                    READ(inputValue,*) abundFile
                    abundFile = trim(abundFile)
                    open(7,file=abundFile,status='unknown')
                CASE('outputFile')
                    READ(inputValue,*) outFile
                    outputFile = trim(outFile)
                    open(10,file=outputFile,status='unknown')
                CASE('columnFile')
                    IF (trim(outSpeciesIn) .NE. '' ) THEN
                        READ(inputValue,*) columnFile
                        columnFile = trim(columnFile)
                        open(11,file=columnFile,status='unknown')
                    ELSE
                        WRITE(*,*) "Error in output species. No species were given but a column file was given."
                        WRITE(*,*) "columnated output requires output species to be chosen."
                        STOP
                    END IF
                CASE('finalOnly')
                    READ(inputValue,*) finalOnly
                    finalOnly = trim(finalOnly)
                CASE DEFAULT
                    WRITE(*,*) "Problem with given parameter:'", trim(inputParameter),"', not supported yet, or invalid"
            END SELECT
        END DO

        IF (readAbunds == 1.0) THEN
            readAbundsBool = .True.
            readAbunds = 0
            tempAV(1) = parameterArray(1, 1, 4)
            temph2form = parameterArray(1, 1, 7)
            tempfc = parameterArray(1, 1, 8)
            tempfo = parameterArray(1, 1, 9)
            tempfmg = parameterArray(1, 1, 10)
            tempfhe = parameterArray(1, 1, 11)
            tempdstep = parameterArray(1, 1, 12)
            !abund(nspec+1,1)=initialDens
            parameterArray = 0.
        ELSE
            readAbundsBool = .False.
        END IF

        CALL initializePhysics
        CALL initializeChemistry

        IF (readAbundsBool.eqv..TRUE.) THEN
            abund(:,1) = allChemicalAbunArray(1,1,:)
            av(1) = tempAV(1)
            h2form = temph2form
            fc = tempfc
            fo = tempfo
            fmg = tempfmg
            fhe = tempfhe
            dstep = tempdstep
            !abund(nspec+1,1)=initialDens
        END IF
        dstep=1
        currentTime=0.0
        timeInYears=0.0

        stepCount(1) = 0
        itterationInteger = 1
        writeCounter = 0

        !loop until the end condition of the model is reached
        DO WHILE ((switch .eq. 1 .and. density(1) < finalDens) .or. (switch .eq. 0 .and. timeInYears < finalTime))
            !store current time as starting point for each depth step
            currentTimeold=currentTime

            !Each physics module has a subroutine to set the target time from the current time
            CALL updateTargetTime

            !loop over parcels, counting from centre out to edge of cloud
            DO dstep=1,points
                !update chemistry from currentTime to targetTime
                CALL updateChemistry

                currentTime=targetTime
                !get time in years for output, currentTime is now equal to targetTime
                timeInYears= currentTime/SECONDS_PER_YEAR

                !Update physics so it's correct for new currentTime and start of next time step
                CALL updatePhysics
                !Sublimation checks if Sublimation should happen this time step and does it
                CALL sublimation(abund)
                !write this depth step now time, chemistry and physics are consistent
                CALL output

                !reset time for next depth point
                if (points .gt. 1)currentTime=currentTimeold

                IF (finalOnly=="False") THEN
                    IF (writeCounter.GE.writeStep) THEN
                        stepCount(1) = stepCount(1) + 1
                        parameterArray(itterationInteger, dstep, 1) = timeInYears
                        parameterArray(itterationInteger, dstep, 2) = density(dstep)
                        parameterArray(itterationInteger, dstep, 3) = temp(dstep)
                        parameterArray(itterationInteger, dstep, 4) = av(dstep)
                        parameterArray(itterationInteger, dstep, 5) = radfield
                        parameterArray(itterationInteger, dstep, 6) = zeta
                        parameterArray(itterationInteger, dstep, 7) = h2form
                        parameterArray(itterationInteger, dstep, 8) = fc
                        parameterArray(itterationInteger, dstep, 9) = fo
                        parameterArray(itterationInteger, dstep, 10) = fmg
                        parameterArray(itterationInteger, dstep, 11) = fhe
                        parameterArray(itterationInteger, dstep, 12) = dstep

                        DO i=1, nspec
                            allChemicalAbunArray(itterationInteger, dstep, i) = abund(i,dstep)
                            DO j=1, nout
                                IF(specname(i).EQ.chemicals(j)) THEN
                                    chemicalAbunArray(itterationInteger, dstep, j) = abund(i,dstep)
                                END IF
                            END DO
                        END DO
                        itterationInteger = itterationInteger + 1
                         writeCounter=1
                    ELSE
                        writeCounter=writeCounter+1
                    END IF
                END IF

                IF((finalOnly=="True").and.&
                        ((switch.eq.1.and.density(1).ge.finalDens).or.&
                                (switch.eq.0.and.timeInYears.ge.finalTime)))THEN
                    stepCount(1) = stepCount(1) + 1
                    parameterArray(itterationInteger, dstep, 1) = timeInYears
                    parameterArray(itterationInteger, dstep, 2) = density(dstep)
                    parameterArray(itterationInteger, dstep, 3) = temp(dstep)
                    parameterArray(itterationInteger, dstep, 4) = av(dstep)
                    parameterArray(itterationInteger, dstep, 5) = radfield
                    parameterArray(itterationInteger, dstep, 6) = zeta
                    parameterArray(itterationInteger, dstep, 7) = h2form
                    parameterArray(itterationInteger, dstep, 8) = fc
                    parameterArray(itterationInteger, dstep, 9) = fo
                    parameterArray(itterationInteger, dstep, 10) = fmg
                    parameterArray(itterationInteger, dstep, 11) = fhe
                    parameterArray(itterationInteger, dstep, 12) = dstep
                    DO i=1, nspec
                        allChemicalAbunArray(itterationInteger, dstep, i) = abund(i,dstep)
                        DO j=1, nout
                            IF(specname(i).EQ.chemicals(j)) THEN
                                chemicalAbunArray(itterationInteger, dstep, j) = abund(i,dstep)
                            END IF
                        END DO
                    END DO
                END IF
            END DO
        END DO
        writeCounter = 0
    END SUBROUTINE TO_DF
END MODULE wrap
