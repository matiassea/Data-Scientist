Attribute VB_Name = "Módulo7"
Sub Create_Pivot_Table_Contrato()
'Todas las lineas de solicitudes que:
'Catalogos y contratos
'No COE
'Oc no realizada


Mensaje = MsgBox("Se creara tabla dinamica" & vbCrLf & "Debe tener actualizada y procesada planilla PO y Ayer" & vbCrLf & "Mostrara las lineas: Catalogo y Contrato" & vbCrLf & "Pais: Chile y Peru" & vbCrLf & "No considera el COE" & vbCrLf & "No considera las lineas con OC compra realizada (filtro Compra realizada)" & vbCrLf & "Cantidad de lineas: se filtra 1 que indica la cantidad de OC", vbOKCancel)
If Mensaje = vbCancel Then
Exit Sub
End If


'Call compra_realizada



'https://www.thespreadsheetguru.com/blog/2014/9/27/vba-guide-excel-pivot-tables
'http://www.globaliconnect.com/excel/index.php?option=com_content&view=article&id=150:excel-pivot-tables-filter-data-items-values-a-dates-using-vba&catid=79&Itemid=475

'Declare Variables
Dim PSheet As Worksheet 'To create a sheet for a new pivot table.
Dim DSheet As Worksheet 'To use as a data sheet.
Dim PCache As PivotCache 'To use as a name for pivot table cache.
Dim PTable As PivotTable 'To use as a name for our pivot table.
Dim PRange As Range 'to define source data range.
Dim lastRow As Long 'To get the last row and column of our data range.
Dim LastCol As Long 'To get the last row and column of our data range.

'Insert a New Blank Worksheet
On Error Resume Next
Application.DisplayAlerts = False
Worksheets("Contrato").Delete
Sheets.Add Before:=ActiveSheet
ActiveSheet.Name = "Contrato"
Application.DisplayAlerts = True
Set PSheet = Worksheets("Contrato")
Set DSheet = Worksheets("Base")

'Define Data Range
lastRow = DSheet.Cells(Rows.Count, 1).End(xlUp).Row
LastCol = DSheet.Cells(5, Columns.Count).End(xlToLeft).Column
Set PRange = DSheet.Cells(5, 1).Resize(lastRow, LastCol)

'Define Pivot Cache
Set PCache = ActiveWorkbook.PivotCaches.Create _
(SourceType:=xlDatabase, SourceData:=PRange). _
CreatePivotTable(TableDestination:=PSheet.Cells(2, 2), _
TableName:="Contrato")

'Insert Blank Pivot Table
Set PTable = PCache.CreatePivotTable _
(TableDestination:=PSheet.Cells(1, 1), TableName:="Contrato")

'Insert Report Filter (estos son los filtros)

With ActiveSheet.PivotTables("Contrato").PivotFields("Cantidad de lineas")
.Orientation = xlPageField
.PivotItems("1").Visible = True
        For i = 2 To .PivotItems.Count
            With .PivotItems(i)
                If i > 1 Then
                    .Visible = False
                End If
            End With
        Next i
.Position = 5
End With

With ActiveSheet.PivotTables("Contrato").PivotFields("Compra realizada")
.Orientation = xlPageField
.PivotItems("OC No Realizada").Visible = True
.PivotItems("OC Realizada").Visible = False
.PivotItems("(blank)").Visible = False
.Position = 4
End With

With ActiveSheet.PivotTables("Contrato").PivotFields("Pais")
.Orientation = xlPageField
.PivotItems("Chile").Visible = True
.PivotItems("Perú").Visible = True
.Position = 3
End With

With ActiveSheet.PivotTables("Contrato").PivotFields("Tipo de compra")
.Orientation = xlPageField
.PivotItems("Catalogo").Visible = False
.PivotItems("Contrato").Visible = True
.PivotItems("Sourcing").Visible = False
.PivotItems("Politica").Visible = False
.PivotItems("(blank)").Visible = False
.Position = 2
End With

With ActiveSheet.PivotTables("Contrato").PivotFields("Area de compra")
.Orientation = xlPageField
.PivotItems("COE").Visible = False
.PivotItems("Compra").Visible = False
.PivotItems("Gestion PO").Visible = True
.PivotItems("(blank)").Visible = False
.Position = 1
End With


'Insert Column Fields  (estos son las columnas)
With ActiveSheet.PivotTables("Contrato").PivotFields("Dias Pen")
.Orientation = xlColumnField
.Position = 1
.PivotItems("(blank)").Visible = False
.Subtotals(1) = False
End With

'Insert Row Labels (estos son las filas)
With ActiveSheet.PivotTables("Contrato").PivotFields("Taxonomia")
.Orientation = xlRowField
.PivotItems("").Visible = False
.PivotItems("(blank)").Visible = False
.PivotItems("OC Realizada").Visible = False
.Position = 1
.Subtotals(1) = False
End With

'Insert Data Field (estos son las valores)
With ActiveSheet.PivotTables("Contrato").PivotFields("Lineadistribucion")
.Orientation = xlDataField
.Position = 1
.Function = xlCount
.NumberFormat = "#,##0"
End With

'http://www.globaliconnect.com/excel/index.php?option=com_content&view=article&id=150:excel-pivot-tables-filter-data-items-values-a-dates-using-vba&catid=79&Itemid=475

'Format Pivot Table
'ActiveSheet.PivotTables("SalesPivotTable").ShowTableStyleRowStripes = True
'ActiveSheet.PivotTables("SalesPivotTable").TableStyle2 = "PivotStyleMedium9"

End Sub

